using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
public class IntegratedGradientsExplainer<T> : ILocalExplainer<T, IntegratedGradientsExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _gradientFunction;
    private readonly int _numFeatures;
    private readonly int _numSteps;
    private readonly Vector<T>? _baseline;
    private readonly string[]? _featureNames;

    /// <inheritdoc/>
    public string MethodName => "IntegratedGradients";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

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
    public IntegratedGradientsExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.");

        // Use zero baseline if none provided
        var baseline = _baseline ?? new Vector<T>(_numFeatures);

        // Compute predictions at input and baseline
        var inputPred = _predictFunction(instance);
        var baselinePred = _predictFunction(baseline);

        double inputPredVal = outputIndex < inputPred.Length ? NumOps.ToDouble(inputPred[outputIndex]) : 0;
        double baselinePredVal = outputIndex < baselinePred.Length ? NumOps.ToDouble(baselinePred[outputIndex]) : 0;
        double predDiff = inputPredVal - baselinePredVal;

        // Compute integrated gradients using Riemann sum approximation
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

        // Compute gradients at each step and integrate
        for (int step = 0; step < _numSteps; step++)
        {
            var gradient = ComputeGradient(scaledInputs[step], outputIndex);

            // Add to running sum (trapezoidal rule)
            double weight = (step == 0 || step == _numSteps - 1) ? 0.5 : 1.0;

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

        // Compute convergence delta (how close attributions sum to prediction difference)
        double attrSum = attributions.Sum(a => NumOps.ToDouble(a));
        double convergenceDelta = Math.Abs(attrSum - predDiff);

        return new IntegratedGradientsExplanation<T>
        {
            Attributions = new Vector<T>(attributions),
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
    public IntegratedGradientsExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new IntegratedGradientsExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
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

            double plusVal = outputIndex < predPlus.Length ? NumOps.ToDouble(predPlus[outputIndex]) : 0;
            double minusVal = outputIndex < predMinus.Length ? NumOps.ToDouble(predMinus[outputIndex]) : 0;

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
    public T BaselinePrediction { get; set; } = default!;

    /// <summary>
    /// Gets or sets the prediction at the input.
    /// </summary>
    public T InputPrediction { get; set; } = default!;

    /// <summary>
    /// Gets or sets the convergence delta (difference between sum of attributions and prediction difference).
    /// A small delta indicates good approximation.
    /// </summary>
    public T ConvergenceDelta { get; set; } = default!;

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
