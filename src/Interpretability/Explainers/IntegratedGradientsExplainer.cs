using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
public class IntegratedGradientsExplainer<T> : ILocalExplainer<T, IntegratedGradientsExplanation<T>>, IGlobalAttributionExplainer<T>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    [Scratch]
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
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
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
            var output = neuralNetwork.Predict(ExplainerTensorOps<T>.RowTensor(input));
            return output.ToVector();
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

        T inputPredVal = inputPred[outputIndex];
        T baselinePredVal = baselinePred[outputIndex];
        T predDiff = NumOps.Subtract(inputPredVal, baselinePredVal);

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
            // Create interpolated inputs along the path
            var scaledInputs = ExplainerTensorOps<T>.InterpolationPath(
                baseline, instance, ExplainerTensorOps<T>.UniformAlphasWithEndpoints(_numSteps));
            var gradients = new Vector<T>[_numSteps + 1];

            // Compute gradients at each step and integrate using trapezoidal rule
            // We need to include both endpoints (step 0 and step _numSteps)
            for (int step = 0; step <= _numSteps; step++)
            {
                var gradient = ComputeGradient(scaledInputs.GetSlice(step).ToVector(), outputIndex);

                // Validate gradient length matches expected number of features
                if (gradient.Length != _numFeatures)
                    throw new InvalidOperationException($"Gradient function returned {gradient.Length} values but expected {_numFeatures}.");

                gradients[step] = gradient;
            }

            // Trapezoidal rule (endpoints get half weight), then multiply by (input - baseline)
            var engine = AiDotNetEngine.Current;
            attributionsVector = engine.Multiply(ExplainerTensorOps<T>.TrapezoidAverage(gradients),
                engine.Subtract(instance, baseline));
        }

        // Compute convergence delta (how close attributions sum to prediction difference)
        T attrSum = AiDotNetEngine.Current.Sum(attributionsVector);
        T convergenceDelta = NumOps.Abs(NumOps.Subtract(attrSum, predDiff));

        return new IntegratedGradientsExplanation<T>
        {
            Attributions = attributionsVector,
            Baseline = baseline,
            Input = instance,
            BaselinePrediction = baselinePredVal,
            InputPrediction = inputPredVal,
            ConvergenceDelta = convergenceDelta,
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
    /// <inheritdoc />
    public Vector<T> ComputeGlobalAttributions(Matrix<T> data)
        => AttributionAggregation.MeanAbsolute<T>(ExplainBatch(data));

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
        return ExplainerTensorOps<T>.CentralDifferenceGradient(_predictFunction, input, outputIndex, 1e-4, _numFeatures);
    }
}

/// <summary>
/// Represents the result of an Integrated Gradients analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class IntegratedGradientsExplanation<T> : AiDotNet.Interpretability.IFeatureAttribution<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the feature attributions.
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <inheritdoc />
    public Vector<T> GetFeatureAttributions() => Attributions;

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
            $"  Baseline prediction: {BaselinePrediction:F4}",
            $"  Input prediction: {InputPrediction:F4}",
            $"  Prediction difference: {NumOps.Subtract(InputPrediction, BaselinePrediction):F4}",
            $"  Convergence delta: {ConvergenceDelta:F6} (smaller is better)",
            "",
            "Top Feature Attributions:"
        };

        var sorted = GetSortedAttributions().Take(10).ToList();
        var signs = ExplainerTensorOps<T>.SignPrefixes(sorted.Select(entry => entry.attribution));
        foreach (var ((name, attr), sign) in sorted.Zip(signs, (entry, prefix) => (entry, prefix)))
        {
            lines.Add($"  {name}: {sign}{attr:F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
