using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// GradientSHAP explainer - a faster approximation of SHAP using gradients.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GradientSHAP combines ideas from Integrated Gradients and SHAP
/// to create a faster method for explaining neural networks.
///
/// How it works:
/// 1. Sample random baselines from your background data
/// 2. For each baseline, compute something like Integrated Gradients
/// 3. Average the results to get SHAP-like values
///
/// Comparison with other methods:
/// - <b>KernelSHAP</b>: Model-agnostic but slow (doesn't use gradients)
/// - <b>DeepSHAP</b>: Fast but requires access to layer activations
/// - <b>GradientSHAP</b>: Good balance - uses gradients but doesn't need internal model access
///
/// Why use GradientSHAP?
/// - Much faster than KernelSHAP for neural networks
/// - Only requires gradient computation (not layer access)
/// - Approximates SHAP values reasonably well
/// - Better than plain gradients (uses baselines for context)
///
/// When to use:
/// - You have a neural network
/// - KernelSHAP is too slow
/// - You don't have access to model internals (for DeepSHAP)
/// </para>
/// </remarks>
public class GradientSHAPExplainer<T> : ILocalExplainer<T, GradientSHAPExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _gradientFunction;
    private readonly Matrix<T> _backgroundData;
    private readonly int _numSamples;
    private readonly int _numSteps;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private readonly bool _addNoise;
    private readonly double _noiseStdDev;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "GradientSHAP";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, GradientSHAP computes
    /// all baseline samples and their gradients in parallel, providing significant speedup.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this method to enable GPU acceleration for gradient computation.
    /// </para>
    /// </remarks>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new GradientSHAP explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions (single sample).</param>
    /// <param name="gradientFunction">Optional function that computes gradients.
    /// Takes (input, outputIndex) and returns gradients. If null, numerical gradients are used.</param>
    /// <param name="backgroundData">Reference dataset for sampling baselines.</param>
    /// <param name="numSamples">Number of baseline samples to use (default: 200).</param>
    /// <param name="numSteps">Number of integration steps (default: 50).</param>
    /// <param name="addNoise">Whether to add noise for smoothing (default: true).</param>
    /// <param name="noiseStdDev">Standard deviation of noise (default: 0.09).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>backgroundData</b>: Sample of your training data. The attributions measure
    ///   importance relative to this "background" distribution.
    /// - <b>numSamples</b>: More samples = more accurate but slower. 200 is usually good.
    /// - <b>addNoise</b>: Small noise helps smooth the attributions (called "SmoothGrad").
    /// </para>
    /// </remarks>
    public GradientSHAPExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, int, Vector<T>>? gradientFunction,
        Matrix<T> backgroundData,
        int numSamples = 200,
        int numSteps = 50,
        bool addNoise = true,
        double noiseStdDev = 0.09,
        string[]? featureNames = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _gradientFunction = gradientFunction;
        _backgroundData = backgroundData ?? throw new ArgumentNullException(nameof(backgroundData));

        if (backgroundData.Rows == 0)
            throw new ArgumentException("Background data must have at least one row.", nameof(backgroundData));
        if (numSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1.", nameof(numSamples));
        if (numSteps < 1)
            throw new ArgumentException("Number of steps must be at least 1.", nameof(numSteps));
        if (addNoise && noiseStdDev <= 0)
            throw new ArgumentException("Noise standard deviation must be positive when addNoise is enabled.", nameof(noiseStdDev));
        if (featureNames != null && featureNames.Length != backgroundData.Columns)
            throw new ArgumentException($"Feature names length ({featureNames.Length}) must match background data columns ({backgroundData.Columns}).", nameof(featureNames));

        _numSamples = numSamples;
        _numSteps = numSteps;
        _addNoise = addNoise;
        _noiseStdDev = noiseStdDev;
        _featureNames = featureNames;
        _randomState = randomState;
    }

    /// <summary>
    /// Computes GradientSHAP attributions for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>GradientSHAP explanation with feature attributions.</returns>
    public GradientSHAPExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: 0);
    }

    /// <summary>
    /// Computes GradientSHAP attributions for a specific output.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">Index of the output to explain.</param>
    /// <returns>GradientSHAP explanation with feature attributions.</returns>
    public GradientSHAPExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        if (instance == null)
            throw new ArgumentNullException(nameof(instance));
        if (outputIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(outputIndex), "Output index cannot be negative.");
        if (instance.Length != _backgroundData.Columns)
            throw new ArgumentException($"Instance length ({instance.Length}) must match background data columns ({_backgroundData.Columns}).", nameof(instance));

        int numFeatures = instance.Length;
        int numBackground = _backgroundData.Rows;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Accumulate attributions
        var attributions = new double[numFeatures];
        var attributionVariances = new double[numFeatures];
        var sampleAttributions = new List<double[]>();

        // Get prediction for input
        var inputPred = _predictFunction(instance);
        if (outputIndex >= inputPred.Length)
            throw new ArgumentOutOfRangeException(nameof(outputIndex), $"Output index {outputIndex} is out of bounds for prediction with {inputPred.Length} outputs.");
        double inputVal = NumOps.ToDouble(inputPred[outputIndex]);

        // Sample baselines and compute attributions
        for (int s = 0; s < _numSamples; s++)
        {
            // Random baseline from background data
            int baselineIdx = rand.Next(numBackground);
            var baseline = _backgroundData.GetRow(baselineIdx);

            // Get baseline prediction
            var baselinePred = _predictFunction(baseline);
            double baselineVal = NumOps.ToDouble(baselinePred[outputIndex]);

            // Compute Integrated Gradients from this baseline
            var sampleAttr = ComputeIntegratedGradients(instance, baseline, outputIndex, rand);

            // Store for variance computation
            var sampleAttrDouble = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++)
            {
                sampleAttrDouble[j] = NumOps.ToDouble(sampleAttr[j]);
                attributions[j] += sampleAttrDouble[j];
            }
            sampleAttributions.Add(sampleAttrDouble);
        }

        // Average attributions
        for (int j = 0; j < numFeatures; j++)
        {
            attributions[j] /= _numSamples;
        }

        // Compute variances
        foreach (var sample in sampleAttributions)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                double diff = sample[j] - attributions[j];
                attributionVariances[j] += diff * diff;
            }
        }
        for (int j = 0; j < numFeatures; j++)
        {
            attributionVariances[j] /= _numSamples;
        }

        // Compute expected value (average prediction over background)
        double expectedValue = 0;
        int bgSampleCount = Math.Min(100, numBackground);
        for (int i = 0; i < bgSampleCount; i++)
        {
            var bg = _backgroundData.GetRow(i);
            var pred = _predictFunction(bg);
            if (outputIndex >= pred.Length)
                throw new InvalidOperationException($"Background prediction {i} has {pred.Length} outputs but outputIndex is {outputIndex}.");
            expectedValue += NumOps.ToDouble(pred[outputIndex]);
        }
        expectedValue /= bgSampleCount;

        // Convert to Vector<T>
        var attrVector = new T[numFeatures];
        var varVector = new T[numFeatures];
        for (int j = 0; j < numFeatures; j++)
        {
            attrVector[j] = NumOps.FromDouble(attributions[j]);
            varVector[j] = NumOps.FromDouble(attributionVariances[j]);
        }

        return new GradientSHAPExplanation<T>
        {
            Attributions = new Vector<T>(attrVector),
            AttributionVariances = new Vector<T>(varVector),
            ExpectedValue = NumOps.FromDouble(expectedValue),
            InputPrediction = NumOps.FromDouble(inputVal),
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray(),
            OutputIndex = outputIndex,
            NumSamples = _numSamples
        };
    }

    /// <inheritdoc/>
    public GradientSHAPExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        if (instances == null)
            throw new ArgumentNullException(nameof(instances));

        var explanations = new GradientSHAPExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes Integrated Gradients from baseline to input.
    /// </summary>
    private T[] ComputeIntegratedGradients(Vector<T> input, Vector<T> baseline, int outputIndex, Random rand)
    {
        int numFeatures = input.Length;
        var attributions = new double[numFeatures];

        // Create interpolated points and sum gradients
        for (int step = 0; step < _numSteps; step++)
        {
            // Random alpha for stochastic integration
            double alpha = (step + rand.NextDouble()) / _numSteps;

            // Interpolate between baseline and input
            var interpolated = new T[numFeatures];
            for (int j = 0; j < numFeatures; j++)
            {
                double baseVal = NumOps.ToDouble(baseline[j]);
                double inputVal = NumOps.ToDouble(input[j]);
                double interpVal = baseVal + alpha * (inputVal - baseVal);

                // Add noise if enabled
                if (_addNoise)
                {
                    interpVal += rand.NextGaussian() * _noiseStdDev;
                }

                interpolated[j] = NumOps.FromDouble(interpVal);
            }

            // Compute gradient at this point
            var gradient = ComputeGradient(new Vector<T>(interpolated), outputIndex);

            // Accumulate
            for (int j = 0; j < numFeatures; j++)
            {
                attributions[j] += NumOps.ToDouble(gradient[j]);
            }
        }

        // Multiply by (input - baseline) and normalize
        var result = new T[numFeatures];
        for (int j = 0; j < numFeatures; j++)
        {
            double deltaInput = NumOps.ToDouble(input[j]) - NumOps.ToDouble(baseline[j]);
            result[j] = NumOps.FromDouble(attributions[j] * deltaInput / _numSteps);
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

        for (int j = 0; j < n; j++)
        {
            var inputPlus = new T[n];
            var inputMinus = new T[n];
            for (int k = 0; k < n; k++)
            {
                inputPlus[k] = input[k];
                inputMinus[k] = input[k];
            }

            inputPlus[j] = NumOps.FromDouble(NumOps.ToDouble(input[j]) + epsilon);
            inputMinus[j] = NumOps.FromDouble(NumOps.ToDouble(input[j]) - epsilon);

            var predPlus = _predictFunction(new Vector<T>(inputPlus));
            var predMinus = _predictFunction(new Vector<T>(inputMinus));

            double valPlus = NumOps.ToDouble(predPlus[outputIndex]);
            double valMinus = NumOps.ToDouble(predMinus[outputIndex]);

            gradient[j] = NumOps.FromDouble((valPlus - valMinus) / (2 * epsilon));
        }

        return new Vector<T>(gradient);
    }
}

/// <summary>
/// Extension methods for random Gaussian sampling.
/// </summary>
internal static class RandomExtensions
{
    /// <summary>
    /// Generates a random number from a standard normal distribution.
    /// </summary>
    public static double NextGaussian(this Random random)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}

/// <summary>
/// Represents the result of a GradientSHAP analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GradientSHAPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the SHAP values (feature attributions).
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the variance of attributions across samples.
    /// High variance indicates uncertainty in the attribution.
    /// </summary>
    public Vector<T> AttributionVariances { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the expected value (average prediction over background data).
    /// </summary>
    public T ExpectedValue { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the prediction for the input.
    /// </summary>
    public T InputPrediction { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the number of samples used.
    /// </summary>
    public int NumSamples { get; set; }

    /// <summary>
    /// Gets attributions sorted by absolute value (most important first).
    /// </summary>
    public List<(string name, T attribution, T variance)> GetSortedAttributions()
    {
        var result = new List<(string, T, T)>();
        for (int i = 0; i < Attributions.Length; i++)
        {
            result.Add((FeatureNames[i], Attributions[i], AttributionVariances[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item2))).ToList();
    }

    /// <summary>
    /// Gets confidence intervals for attributions.
    /// </summary>
    /// <param name="numStdDev">Number of standard deviations for interval (default: 2).</param>
    public List<(string name, T lower, T attribution, T upper)> GetAttributionsWithConfidence(double numStdDev = 2)
    {
        var result = new List<(string, T, T, T)>();
        for (int i = 0; i < Attributions.Length; i++)
        {
            double attr = NumOps.ToDouble(Attributions[i]);
            double std = Math.Sqrt(NumOps.ToDouble(AttributionVariances[i]));
            result.Add((
                FeatureNames[i],
                NumOps.FromDouble(attr - numStdDev * std),
                Attributions[i],
                NumOps.FromDouble(attr + numStdDev * std)
            ));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item3))).ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        double attrSum = 0;
        for (int i = 0; i < Attributions.Length; i++)
        {
            attrSum += NumOps.ToDouble(Attributions[i]);
        }

        var lines = new List<string>
        {
            "GradientSHAP Explanation:",
            $"  Expected value (base): {NumOps.ToDouble(ExpectedValue):F4}",
            $"  Input prediction: {NumOps.ToDouble(InputPrediction):F4}",
            $"  Sum of SHAP values: {attrSum:F4}",
            $"  Samples used: {NumSamples}",
            "",
            "Top Feature Attributions (with ±2σ confidence):"
        };

        var withConfidence = GetAttributionsWithConfidence().Take(10);
        foreach (var (name, lower, attr, upper) in withConfidence)
        {
            double attrVal = NumOps.ToDouble(attr);
            double lowerVal = NumOps.ToDouble(lower);
            double upperVal = NumOps.ToDouble(upper);
            string sign = attrVal >= 0 ? "+" : "";
            lines.Add($"  {name}: {sign}{attrVal:F4} [{lowerVal:F4}, {upperVal:F4}]");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
