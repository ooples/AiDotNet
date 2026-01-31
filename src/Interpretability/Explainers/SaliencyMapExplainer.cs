using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Saliency Map explainer using gradient-based methods.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Saliency maps are one of the simplest ways to explain neural networks.
/// They show which input features are most "sensitive" - where small changes would most affect the output.
///
/// Types of saliency methods:
/// 1. <b>Vanilla Gradient</b>: The raw gradient of output w.r.t. input
/// 2. <b>Gradient × Input</b>: Gradient multiplied by input (more interpretable)
/// 3. <b>SmoothGrad</b>: Average gradient over noisy versions (reduces noise)
/// 4. <b>Guided Backpropagation</b>: Only propagates positive gradients
///
/// How to interpret:
/// - High absolute saliency = changing this feature would change the output a lot
/// - For images: bright spots show important pixels
/// - For tabular data: high values show important features
///
/// Pros:
/// - Fast to compute (single backward pass)
/// - Easy to understand
/// - Works with any differentiable model
///
/// Cons:
/// - Can be noisy (especially vanilla gradient)
/// - Doesn't show actual contribution, just sensitivity
/// - Can miss important features with low gradient
///
/// SmoothGrad is recommended for cleaner visualizations.
/// </para>
/// </remarks>
public class SaliencyMapExplainer<T> : ILocalExplainer<T, SaliencyMapExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _gradientFunction;
    private readonly int _numFeatures;
    private readonly SaliencyMethod _method;
    private readonly int _smoothGradSamples;
    private readonly double _smoothGradNoise;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;

    /// <inheritdoc/>
    public string MethodName => $"SaliencyMap ({_method})";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Saliency Map explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="gradientFunction">Optional function that computes gradients.
    /// If null, numerical gradients will be computed.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="method">Saliency method to use (default: GradientTimesInput).</param>
    /// <param name="smoothGradSamples">Number of samples for SmoothGrad (default: 50).</param>
    /// <param name="smoothGradNoise">Noise standard deviation for SmoothGrad (default: 0.1).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>method</b>: GradientTimesInput is the most commonly used and interpretable
    /// - <b>smoothGradSamples</b>: More samples = smoother but slower. 50 is a good default.
    /// - <b>smoothGradNoise</b>: How much random noise to add. 0.1-0.2 works well.
    /// </para>
    /// </remarks>
    public SaliencyMapExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, int, Vector<T>>? gradientFunction = null,
        int numFeatures = 0,
        SaliencyMethod method = SaliencyMethod.GradientTimesInput,
        int smoothGradSamples = 50,
        double smoothGradNoise = 0.1,
        string[]? featureNames = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (smoothGradSamples < 1)
            throw new ArgumentOutOfRangeException(nameof(smoothGradSamples), "SmoothGrad samples must be at least 1.");
        if (smoothGradNoise < 0)
            throw new ArgumentOutOfRangeException(nameof(smoothGradNoise), "SmoothGrad noise must be non-negative.");
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match numFeatures ({numFeatures}).", nameof(featureNames));

        _gradientFunction = gradientFunction;
        _numFeatures = numFeatures;
        _method = method;
        _smoothGradSamples = smoothGradSamples;
        _smoothGradNoise = smoothGradNoise;
        _featureNames = featureNames;
        _randomState = randomState;
    }

    /// <summary>
    /// Computes saliency map for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>Saliency map explanation.</returns>
    public SaliencyMapExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: -1);
    }

    /// <summary>
    /// Computes saliency map for a specific output.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">Index of the output to explain (-1 for highest scoring).</param>
    /// <returns>Saliency map explanation.</returns>
    public SaliencyMapExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        int numFeatures = instance.Length;

        // Get prediction and determine target output
        var prediction = _predictFunction(instance);
        if (outputIndex < 0)
        {
            outputIndex = GetArgMax(prediction);
        }

        T[] saliency;

        switch (_method)
        {
            case SaliencyMethod.VanillaGradient:
                saliency = ComputeVanillaGradient(instance, outputIndex);
                break;
            case SaliencyMethod.GradientTimesInput:
                saliency = ComputeGradientTimesInput(instance, outputIndex);
                break;
            case SaliencyMethod.SmoothGrad:
                saliency = ComputeSmoothGrad(instance, outputIndex);
                break;
            case SaliencyMethod.SmoothGradSquared:
                saliency = ComputeSmoothGradSquared(instance, outputIndex);
                break;
            default:
                saliency = ComputeGradientTimesInput(instance, outputIndex);
                break;
        }

        // Compute statistics
        double maxAbs = 0;
        double sumAbs = 0;
        for (int i = 0; i < numFeatures; i++)
        {
            double absVal = Math.Abs(NumOps.ToDouble(saliency[i]));
            maxAbs = Math.Max(maxAbs, absVal);
            sumAbs += absVal;
        }

        // Normalize to get relative importance
        var normalizedSaliency = new T[numFeatures];
        for (int i = 0; i < numFeatures; i++)
        {
            normalizedSaliency[i] = maxAbs > 0
                ? NumOps.FromDouble(Math.Abs(NumOps.ToDouble(saliency[i])) / maxAbs)
                : NumOps.Zero;
        }

        return new SaliencyMapExplanation<T>
        {
            Saliency = new Vector<T>(saliency),
            NormalizedSaliency = new Vector<T>(normalizedSaliency),
            Input = instance,
            Prediction = prediction,
            OutputIndex = outputIndex,
            Method = _method,
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray()
        };
    }

    /// <inheritdoc/>
    public SaliencyMapExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new SaliencyMapExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes vanilla gradient.
    /// </summary>
    private T[] ComputeVanillaGradient(Vector<T> input, int outputIndex)
    {
        var gradient = ComputeGradient(input, outputIndex);
        var result = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = gradient[i];
        }
        return result;
    }

    /// <summary>
    /// Computes gradient × input.
    /// </summary>
    private T[] ComputeGradientTimesInput(Vector<T> input, int outputIndex)
    {
        var gradient = ComputeGradient(input, outputIndex);
        var result = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.FromDouble(
                NumOps.ToDouble(gradient[i]) * NumOps.ToDouble(input[i])
            );
        }
        return result;
    }

    /// <summary>
    /// Computes SmoothGrad (average gradient over noisy inputs).
    /// </summary>
    private T[] ComputeSmoothGrad(Vector<T> input, int outputIndex)
    {
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int n = input.Length;
        var sumGradient = new double[n];

        // Compute scale for noise (based on input range)
        double inputRange = 0;
        for (int i = 0; i < n; i++)
        {
            inputRange = Math.Max(inputRange, Math.Abs(NumOps.ToDouble(input[i])));
        }
        double noiseScale = _smoothGradNoise * Math.Max(1, inputRange);

        for (int s = 0; s < _smoothGradSamples; s++)
        {
            // Add noise to input
            var noisyInput = new T[n];
            for (int i = 0; i < n; i++)
            {
                double noise = rand.NextGaussian() * noiseScale;
                noisyInput[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) + noise);
            }

            // Compute gradient for noisy input
            var gradient = ComputeGradient(new Vector<T>(noisyInput), outputIndex);

            // Accumulate
            for (int i = 0; i < n; i++)
            {
                sumGradient[i] += NumOps.ToDouble(gradient[i]);
            }
        }

        // Average and multiply by input
        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            double avgGrad = sumGradient[i] / _smoothGradSamples;
            result[i] = NumOps.FromDouble(avgGrad * NumOps.ToDouble(input[i]));
        }

        return result;
    }

    /// <summary>
    /// Computes SmoothGrad² (squared gradients for sharper focus).
    /// </summary>
    private T[] ComputeSmoothGradSquared(Vector<T> input, int outputIndex)
    {
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int n = input.Length;
        var sumGradientSq = new double[n];

        double inputRange = 0;
        for (int i = 0; i < n; i++)
        {
            inputRange = Math.Max(inputRange, Math.Abs(NumOps.ToDouble(input[i])));
        }
        double noiseScale = _smoothGradNoise * Math.Max(1, inputRange);

        for (int s = 0; s < _smoothGradSamples; s++)
        {
            var noisyInput = new T[n];
            for (int i = 0; i < n; i++)
            {
                double noise = rand.NextGaussian() * noiseScale;
                noisyInput[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) + noise);
            }

            var gradient = ComputeGradient(new Vector<T>(noisyInput), outputIndex);

            for (int i = 0; i < n; i++)
            {
                double g = NumOps.ToDouble(gradient[i]);
                sumGradientSq[i] += g * g;
            }
        }

        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.FromDouble(sumGradientSq[i] / _smoothGradSamples);
        }

        return result;
    }

    /// <summary>
    /// Computes gradient using provided function or numerical approximation.
    /// </summary>
    private Vector<T> ComputeGradient(Vector<T> input, int outputIndex)
    {
        if (_gradientFunction != null)
        {
            return _gradientFunction(input, outputIndex);
        }

        return ComputeNumericalGradient(input, outputIndex);
    }

    /// <summary>
    /// Computes numerical gradient.
    /// </summary>
    private Vector<T> ComputeNumericalGradient(Vector<T> input, int outputIndex)
    {
        int n = input.Length;
        var gradient = new T[n];
        double epsilon = 1e-4;

        for (int i = 0; i < n; i++)
        {
            var inputPlus = new T[n];
            var inputMinus = new T[n];
            for (int j = 0; j < n; j++)
            {
                inputPlus[j] = input[j];
                inputMinus[j] = input[j];
            }

            inputPlus[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) + epsilon);
            inputMinus[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) - epsilon);

            var predPlus = _predictFunction(new Vector<T>(inputPlus));
            var predMinus = _predictFunction(new Vector<T>(inputMinus));

            double valPlus = outputIndex < predPlus.Length ? NumOps.ToDouble(predPlus[outputIndex]) : 0;
            double valMinus = outputIndex < predMinus.Length ? NumOps.ToDouble(predMinus[outputIndex]) : 0;

            gradient[i] = NumOps.FromDouble((valPlus - valMinus) / (2 * epsilon));
        }

        return new Vector<T>(gradient);
    }

    /// <summary>
    /// Gets the index of the maximum value.
    /// </summary>
    private int GetArgMax(Vector<T> values)
    {
        int maxIdx = 0;
        double maxVal = double.MinValue;
        for (int i = 0; i < values.Length; i++)
        {
            double val = NumOps.ToDouble(values[i]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

/// <summary>
/// Saliency computation methods.
/// </summary>
public enum SaliencyMethod
{
    /// <summary>
    /// Raw gradient of output w.r.t. input.
    /// </summary>
    VanillaGradient,

    /// <summary>
    /// Gradient multiplied by input value.
    /// </summary>
    GradientTimesInput,

    /// <summary>
    /// Average gradient over noisy inputs (smoother).
    /// </summary>
    SmoothGrad,

    /// <summary>
    /// Squared average gradient (sharper focus).
    /// </summary>
    SmoothGradSquared
}

/// <summary>
/// Represents the result of a Saliency Map analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SaliencyMapExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the raw saliency values.
    /// </summary>
    public Vector<T> Saliency { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the normalized saliency values (0 to 1).
    /// </summary>
    public Vector<T> NormalizedSaliency { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Input { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the model prediction.
    /// </summary>
    public Vector<T> Prediction { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the saliency method used.
    /// </summary>
    public SaliencyMethod Method { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets saliency values sorted by absolute value (most salient first).
    /// </summary>
    public List<(string name, T saliency, T normalized)> GetSortedSaliency()
    {
        var result = new List<(string, T, T)>();
        for (int i = 0; i < Saliency.Length; i++)
        {
            result.Add((FeatureNames[i], Saliency[i], NormalizedSaliency[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item2))).ToList();
    }

    /// <summary>
    /// Gets the top K most salient features.
    /// </summary>
    public List<(string name, T saliency)> GetTopSalientFeatures(int topK = 10)
    {
        return GetSortedSaliency()
            .Take(topK)
            .Select(x => (x.name, x.saliency))
            .ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            $"Saliency Map Explanation (Method: {Method}):",
            $"  Output index: {OutputIndex}",
            "",
            "Top Salient Features:"
        };

        var top = GetTopSalientFeatures(10);
        foreach (var (name, sal) in top)
        {
            double salVal = NumOps.ToDouble(sal);
            string sign = salVal >= 0 ? "+" : "";
            lines.Add($"  {name}: {sign}{salVal:F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
