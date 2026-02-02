using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Integrated Gradients explainer for neural networks with gradient access.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Integrated Gradients is a method for explaining neural network predictions
/// that satisfies two important mathematical properties:
///
/// 1. <b>Completeness (Axiom of Completeness)</b>: The attributions sum up to the difference
///    between the prediction at the input and the prediction at a baseline (usually zeros).
///
/// 2. <b>Sensitivity</b>: If a feature differs between input and baseline and affects the output,
///    it gets a non-zero attribution.
///
/// How it works:
/// - Start with a "baseline" (typically all zeros or a neutral input)
/// - Create a path from baseline to your actual input
/// - Integrate the gradients along this path
/// - The result shows how much each feature contributed to moving from baseline to final prediction
///
/// Why use Integrated Gradients?
/// - Theoretically sound (satisfies axioms that other methods don't)
/// - Works with any differentiable model
/// - Attributions have clear meaning: contribution to prediction difference from baseline
///
/// Example: For an image classifier predicting "cat":
/// - Baseline: black image (all zeros)
/// - Input: image of a cat
/// - Integrated Gradients shows which pixels contributed most to the "cat" prediction
/// </para>
/// </remarks>
public class IntegratedGradientsExplainer<T> : ILocalExplainer<T, IntegratedGradientsExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _gradientFunction;
    private readonly int _numFeatures;
    private readonly int _numSteps;
    private readonly Vector<T>? _baseline;
    private readonly string[]? _featureNames;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "IntegratedGradients";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, Integrated Gradients computes
    /// all path gradients in parallel, significantly speeding up the attribution computation.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this method to enable GPU acceleration for path integration.
    /// Example:
    /// <code>
    /// var helper = GPUExplainerHelper&lt;double&gt;.CreateWithAutoDetect();
    /// explainer.SetGPUHelper(helper);
    /// </code>
    /// </para>
    /// </remarks>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new Integrated Gradients explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes a single input vector and returns predictions.</param>
    /// <param name="gradientFunction">
    /// A function that computes gradients with respect to input.
    /// Takes (input, outputIndex) and returns gradient vector.
    /// If null, numerical gradients will be computed (slower but works for any model).
    /// </param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numSteps">Number of steps for integration (default: 50).</param>
    /// <param name="baseline">Baseline input (default: zeros).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>numSteps</b>: More steps = more accurate integration, but slower. 50-300 is typical.
    /// - <b>baseline</b>: The "neutral" input to compare against. Zeros work well for most cases.
    ///   For images, you might use a black image or a blurred version of the input.
    /// - <b>gradientFunction</b>: If your model can compute gradients (neural networks), provide it
    ///   for faster computation. Otherwise, numerical gradients will be used.
    /// </para>
    /// </remarks>
    public IntegratedGradientsExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, int, Vector<T>>? gradientFunction,
        int numFeatures,
        int numSteps = 50,
        Vector<T>? baseline = null,
        string[]? featureNames = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _gradientFunction = gradientFunction;

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (numSteps < 2)
            throw new ArgumentException("Number of steps must be at least 2.", nameof(numSteps));
        if (baseline != null && baseline.Length != numFeatures)
            throw new ArgumentException($"Baseline has {baseline.Length} features but expected {numFeatures}.", nameof(baseline));
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"Feature names length ({featureNames.Length}) does not match feature count ({numFeatures}).", nameof(featureNames));

        _numFeatures = numFeatures;
        _numSteps = numSteps;
        _baseline = baseline;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Initializes a new Integrated Gradients explainer from a neural network model.
    /// </summary>
    /// <param name="neuralNetwork">The neural network model with backpropagation support.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numSteps">Number of steps for integration (default: 50).</param>
    /// <param name="baseline">Baseline input (default: zeros).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the preferred constructor when you have a neural network model.
    /// It automatically uses the network's built-in backpropagation to compute exact gradients,
    /// which is much faster and more accurate than numerical approximation.
    ///
    /// The neural network's ForwardWithMemory and Backpropagate methods are used to efficiently
    /// compute input gradients. This is the same technique used during training but applied
    /// for interpretation.
    ///
    /// Benefits over numerical gradients:
    /// - <b>Speed</b>: O(1) forward/backward passes vs O(n) for numerical gradients
    /// - <b>Accuracy</b>: Exact gradients vs approximations with numerical precision issues
    /// - <b>Stability</b>: No issues with choosing epsilon values
    /// </para>
    /// </remarks>
    public IntegratedGradientsExplainer(
        INeuralNetwork<T> neuralNetwork,
        int numFeatures,
        int numSteps = 50,
        Vector<T>? baseline = null,
        string[]? featureNames = null)
    {
        if (neuralNetwork == null)
            throw new ArgumentNullException(nameof(neuralNetwork));
        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (numSteps < 2)
            throw new ArgumentException("Number of steps must be at least 2.", nameof(numSteps));

        // Create gradient helper that uses backpropagation
        var gradientHelper = new InputGradientHelper<T>(neuralNetwork);

        // Create predict function wrapper
        _predictFunction = input =>
        {
            var tensor = new Tensor<T>(new[] { 1, input.Length });
            for (int i = 0; i < input.Length; i++)
            {
                tensor[0, i] = input[i];
            }
            var output = neuralNetwork.Predict(tensor);
            var result = new T[output.Length];
            var outputArray = output.ToArray();
            for (int i = 0; i < output.Length; i++)
            {
                result[i] = outputArray[i];
            }
            return new Vector<T>(result);
        };

        // Use backpropagation-based gradient function
        _gradientFunction = gradientHelper.CreateGradientFunction();

        _numFeatures = numFeatures;
        _numSteps = numSteps;
        _baseline = baseline;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Computes Integrated Gradients attributions for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>Integrated Gradients explanation with feature attributions.</returns>
    public IntegratedGradientsExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: 0);
    }

    /// <summary>
    /// Computes Integrated Gradients attributions for a specific output class.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">The index of the output to explain (for multi-class models).</param>
    /// <returns>Integrated Gradients explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> With GPU acceleration enabled, this method computes all path
    /// gradients in parallel rather than sequentially, providing significant speedup for
    /// large models and high numSteps values.
    /// </para>
    /// </remarks>
    public IntegratedGradientsExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.");

        // Use zero baseline if none provided
        var baseline = _baseline ?? new Vector<T>(_numFeatures);

        // Compute predictions at input and baseline
        var inputPred = _predictFunction(instance);
        var baselinePred = _predictFunction(baseline);

        if (outputIndex < 0 || outputIndex >= inputPred.Length)
            throw new ArgumentOutOfRangeException(nameof(outputIndex), $"Output index {outputIndex} is out of bounds for prediction with {inputPred.Length} outputs.");

        double inputPredVal = NumOps.ToDouble(inputPred[outputIndex]);
        double baselinePredVal = NumOps.ToDouble(baselinePred[outputIndex]);
        double predDiff = inputPredVal - baselinePredVal;

        Vector<T> attributionsVector;

        // Use GPU-accelerated path integration if available
        if (_gpuHelper != null && _gpuHelper.IsGPUEnabled && _gradientFunction != null)
        {
            attributionsVector = _gpuHelper.ComputeIntegratedGradientsParallel(
                _gradientFunction, instance, baseline, _numSteps, outputIndex);
        }
        else
        {
            // Standard sequential computation
            var attributions = new T[_numFeatures];
            var scaledInputs = new Vector<T>[_numSteps + 1];

            // Create interpolated inputs along the path
            for (int step = 0; step <= _numSteps; step++)
            {
                double alpha = (double)step / _numSteps;
                var scaled = new T[_numFeatures];

                for (int j = 0; j < _numFeatures; j++)
                {
                    double baseVal = NumOps.ToDouble(baseline[j]);
                    double inputVal = NumOps.ToDouble(instance[j]);
                    scaled[j] = NumOps.FromDouble(baseVal + alpha * (inputVal - baseVal));
                }
                scaledInputs[step] = new Vector<T>(scaled);
            }

            // Compute gradients at each step and integrate using trapezoidal rule
            // We need to include both endpoints (step 0 and step _numSteps)
            for (int step = 0; step <= _numSteps; step++)
            {
                var gradient = ComputeGradient(scaledInputs[step], outputIndex);

                // Validate gradient length matches expected number of features
                if (gradient.Length != _numFeatures)
                    throw new InvalidOperationException($"Gradient function returned {gradient.Length} values but expected {_numFeatures}.");

                // Add to running sum (trapezoidal rule: endpoints get half weight)
                double weight = (step == 0 || step == _numSteps) ? 0.5 : 1.0;

                for (int j = 0; j < _numFeatures; j++)
                {
                    double gradVal = NumOps.ToDouble(gradient[j]);
                    attributions[j] = NumOps.FromDouble(
                        NumOps.ToDouble(attributions[j]) + weight * gradVal / _numSteps);
                }
            }

            // Multiply by (input - baseline)
            for (int j = 0; j < _numFeatures; j++)
            {
                double inputVal = NumOps.ToDouble(instance[j]);
                double baseVal = NumOps.ToDouble(baseline[j]);
                attributions[j] = NumOps.FromDouble(NumOps.ToDouble(attributions[j]) * (inputVal - baseVal));
            }

            attributionsVector = new Vector<T>(attributions);
        }

        // Compute convergence delta (how close attributions sum to prediction difference)
        double attrSum = 0;
        for (int j = 0; j < attributionsVector.Length; j++)
        {
            attrSum += NumOps.ToDouble(attributionsVector[j]);
        }
        double convergenceDelta = Math.Abs(attrSum - predDiff);

        return new IntegratedGradientsExplanation<T>
        {
            Attributions = attributionsVector,
            Baseline = baseline,
            Input = instance,
            BaselinePrediction = NumOps.FromDouble(baselinePredVal),
            InputPrediction = NumOps.FromDouble(inputPredVal),
            ConvergenceDelta = NumOps.FromDouble(convergenceDelta),
            FeatureNames = _featureNames ?? Enumerable.Range(0, _numFeatures).Select(i => $"Feature {i}").ToArray(),
            OutputIndex = outputIndex,
            NumSteps = _numSteps
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, batch explanations are computed
    /// in parallel across multiple instances, further improving performance.
    /// </para>
    /// </remarks>
    public IntegratedGradientsExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new IntegratedGradientsExplanation<T>[instances.Rows];

        if (_gpuHelper != null && _gpuHelper.IsGPUEnabled && instances.Rows > 1)
        {
            // Use parallel processing for batch explanations
            Parallel.For(0, instances.Rows, new ParallelOptions
            {
                MaxDegreeOfParallelism = _gpuHelper.MaxParallelism
            }, i =>
            {
                explanations[i] = Explain(instances.GetRow(i));
            });
        }
        else
        {
            for (int i = 0; i < instances.Rows; i++)
            {
                explanations[i] = Explain(instances.GetRow(i));
            }
        }

        return explanations;
    }

    /// <summary>
    /// Computes gradient either using provided function or numerical approximation.
    /// </summary>
    private Vector<T> ComputeGradient(Vector<T> input, int outputIndex)
    {
        if (_gradientFunction != null)
        {
            return _gradientFunction(input, outputIndex);
        }

        // Numerical gradient approximation
        return ComputeNumericalGradient(input, outputIndex);
    }

    /// <summary>
    /// Computes numerical gradient using central difference.
    /// </summary>
    private Vector<T> ComputeNumericalGradient(Vector<T> input, int outputIndex)
    {
        var gradient = new T[_numFeatures];
        double epsilon = 1e-4;

        for (int j = 0; j < _numFeatures; j++)
        {
            // Forward perturbation
            var inputPlus = new T[_numFeatures];
            var inputMinus = new T[_numFeatures];

            for (int k = 0; k < _numFeatures; k++)
            {
                inputPlus[k] = input[k];
                inputMinus[k] = input[k];
            }

            inputPlus[j] = NumOps.FromDouble(NumOps.ToDouble(input[j]) + epsilon);
            inputMinus[j] = NumOps.FromDouble(NumOps.ToDouble(input[j]) - epsilon);

            var predPlus = _predictFunction(new Vector<T>(inputPlus));
            var predMinus = _predictFunction(new Vector<T>(inputMinus));

            double plusVal = NumOps.ToDouble(predPlus[outputIndex]);
            double minusVal = NumOps.ToDouble(predMinus[outputIndex]);

            gradient[j] = NumOps.FromDouble((plusVal - minusVal) / (2 * epsilon));
        }

        return new Vector<T>(gradient);
    }
}

/// <summary>
/// Represents the result of an Integrated Gradients analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class IntegratedGradientsExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the feature attributions.
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the baseline used.
    /// </summary>
    public Vector<T> Baseline { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Input { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the prediction at the baseline.
    /// </summary>
    public T BaselinePrediction { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the prediction at the input.
    /// </summary>
    public T InputPrediction { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the convergence delta (difference between sum of attributions and prediction difference).
    /// A small delta indicates good approximation.
    /// </summary>
    public T ConvergenceDelta { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the number of integration steps used.
    /// </summary>
    public int NumSteps { get; set; }

    /// <summary>
    /// Gets attributions sorted by absolute value (most important first).
    /// </summary>
    public List<(string name, T attribution)> GetSortedAttributions()
    {
        var result = new List<(string, T)>();
        for (int i = 0; i < Attributions.Length; i++)
        {
            result.Add((FeatureNames[i], Attributions[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item2))).ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            "Integrated Gradients Explanation:",
            $"  Baseline prediction: {NumOps.ToDouble(BaselinePrediction):F4}",
            $"  Input prediction: {NumOps.ToDouble(InputPrediction):F4}",
            $"  Prediction difference: {NumOps.ToDouble(InputPrediction) - NumOps.ToDouble(BaselinePrediction):F4}",
            $"  Convergence delta: {NumOps.ToDouble(ConvergenceDelta):F6} (smaller is better)",
            "",
            "Top Feature Attributions:"
        };

        var sorted = GetSortedAttributions().Take(10);
        foreach (var (name, attr) in sorted)
        {
            double val = NumOps.ToDouble(attr);
            string sign = val >= 0 ? "+" : "";
            lines.Add($"  {name}: {sign}{val:F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
