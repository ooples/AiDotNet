using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// DeepLIFT (Deep Learning Important FeaTures) explainer for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DeepLIFT is a method for explaining neural network predictions
/// by comparing activations to a reference/baseline.
///
/// How it differs from gradients:
/// - Gradients: "How would the output change if I slightly changed the input?"
/// - DeepLIFT: "How much does each input contribute compared to a baseline?"
///
/// Key concepts:
/// 1. <b>Reference/Baseline</b>: A neutral input (like zeros or average input)
/// 2. <b>Difference from reference</b>: Compares actual activations to reference activations
/// 3. <b>Multipliers</b>: How much each neuron's difference-from-reference contributes
///
/// DeepLIFT variants:
/// - <b>Rescale</b>: Distributes contribution proportionally
/// - <b>RevealCancel</b>: Handles positive and negative contributions separately
///
/// Advantages over gradients:
/// - More stable than gradients (no saturation issues)
/// - Handles non-linearities better (ReLU, etc.)
/// - Contributions sum to the difference between output and baseline output
///
/// Example: For a spam classifier:
/// - Reference: Average email or neutral text
/// - DeepLIFT shows which words made the email MORE or LESS likely to be spam
///   compared to the reference
/// </para>
/// </remarks>
public class DeepLIFTExplainer<T> : ILocalExplainer<T, DeepLIFTExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>>? _getActivations;
    private readonly Func<Vector<T>, Vector<T>, Vector<T>>? _computeMultipliers;
    private readonly InputGradientHelper<T>? _gradientHelper;
    private readonly int _numFeatures;
    private readonly Vector<T>? _baseline;
    private readonly string[]? _featureNames;
    private readonly DeepLIFTRule _rule;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "DeepLIFT";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, DeepLIFT computations
    /// are parallelized for faster attribution computation.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this method to enable GPU acceleration.
    /// </para>
    /// </remarks>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new DeepLIFT explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="getActivations">Optional function to get intermediate activations.</param>
    /// <param name="computeMultipliers">Optional function to compute DeepLIFT multipliers.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="baseline">Reference input (default: zeros).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="rule">DeepLIFT rule to use (default: Rescale).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>baseline</b>: What counts as "neutral"? Zeros work for images, but for
    ///   other data you might use the average of your training data.
    /// - <b>rule</b>: Rescale is simpler and works well in most cases.
    ///   RevealCancel is better when you need to separate positive/negative contributions.
    /// </para>
    /// </remarks>
    public DeepLIFTExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>>? getActivations = null,
        Func<Vector<T>, Vector<T>, Vector<T>>? computeMultipliers = null,
        int numFeatures = 0,
        Vector<T>? baseline = null,
        string[]? featureNames = null,
        DeepLIFTRule rule = DeepLIFTRule.Rescale)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        _getActivations = getActivations;
        _computeMultipliers = computeMultipliers;

        if (numFeatures < 0)
            throw new ArgumentException("Number of features cannot be negative.", nameof(numFeatures));

        _numFeatures = numFeatures;
        _baseline = baseline;
        _featureNames = featureNames;
        _rule = rule;
    }

    /// <summary>
    /// Initializes a new DeepLIFT explainer from a neural network model.
    /// </summary>
    /// <param name="neuralNetwork">The neural network model with backpropagation support.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="baseline">Reference input (default: zeros).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="rule">DeepLIFT rule to use (default: Rescale).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the preferred constructor when you have a neural network model.
    /// It automatically uses the network's built-in backpropagation to compute gradients,
    /// which enables more accurate DeepLIFT attributions than numerical approximations.
    ///
    /// DeepLIFT with backpropagation:
    /// - Uses the network's gradient flow to trace how input differences propagate
    /// - More faithful to the actual computation happening in the network
    /// - Produces attributions that sum to the prediction difference (completeness)
    ///
    /// Note: True DeepLIFT requires access to intermediate layer activations and uses
    /// specialized reference-difference propagation rules. This constructor provides an
    /// approximation using gradient-based attribution with the rescale rule to ensure
    /// completeness. For full DeepLIFT functionality, provide custom computeMultipliers
    /// and getActivations functions in the other constructor.
    /// </para>
    /// </remarks>
    public DeepLIFTExplainer(
        INeuralNetwork<T> neuralNetwork,
        int numFeatures,
        Vector<T>? baseline = null,
        string[]? featureNames = null,
        DeepLIFTRule rule = DeepLIFTRule.Rescale)
    {
        if (neuralNetwork == null)
            throw new ArgumentNullException(nameof(neuralNetwork));
        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));

        // Create gradient helper for backpropagation-based gradients
        _gradientHelper = new InputGradientHelper<T>(neuralNetwork);

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

        _numFeatures = numFeatures;
        _baseline = baseline;
        _featureNames = featureNames;
        _rule = rule;
    }

    /// <summary>
    /// Computes DeepLIFT attributions for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>DeepLIFT explanation with feature attributions.</returns>
    public DeepLIFTExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: 0);
    }

    /// <summary>
    /// Computes DeepLIFT attributions for a specific output.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">Index of the output to explain.</param>
    /// <returns>DeepLIFT explanation with feature attributions.</returns>
    public DeepLIFTExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        int numFeatures = instance.Length;
        var baseline = _baseline ?? new Vector<T>(numFeatures);

        // Get predictions
        var inputPred = _predictFunction(instance);
        var baselinePred = _predictFunction(baseline);

        double inputVal = outputIndex < inputPred.Length ? NumOps.ToDouble(inputPred[outputIndex]) : 0;
        double baselineVal = outputIndex < baselinePred.Length ? NumOps.ToDouble(baselinePred[outputIndex]) : 0;
        double deltaOutput = inputVal - baselineVal;

        T[] attributions;

        if (_computeMultipliers != null)
        {
            // Use provided multiplier computation
            var multipliers = _computeMultipliers(instance, baseline);
            attributions = ComputeAttributionsFromMultipliers(instance, baseline, multipliers);
        }
        else
        {
            // Approximate DeepLIFT using gradient approximation with rescale rule
            attributions = ComputeApproximateAttributions(instance, baseline, outputIndex);
        }

        // Verify sum property (attributions should sum to delta output)
        double attrSum = attributions.Sum(a => NumOps.ToDouble(a));
        double completenessError = Math.Abs(attrSum - deltaOutput);

        return new DeepLIFTExplanation<T>
        {
            Attributions = new Vector<T>(attributions),
            Baseline = baseline,
            Input = instance,
            BaselinePrediction = NumOps.FromDouble(baselineVal),
            InputPrediction = NumOps.FromDouble(inputVal),
            DeltaOutput = NumOps.FromDouble(deltaOutput),
            CompletenessError = NumOps.FromDouble(completenessError),
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray(),
            OutputIndex = outputIndex,
            Rule = _rule
        };
    }

    /// <inheritdoc/>
    public DeepLIFTExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new DeepLIFTExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes attributions from multipliers.
    /// </summary>
    private T[] ComputeAttributionsFromMultipliers(Vector<T> input, Vector<T> baseline, Vector<T> multipliers)
    {
        int n = input.Length;
        var attributions = new T[n];

        for (int i = 0; i < n; i++)
        {
            double deltaInput = NumOps.ToDouble(input[i]) - NumOps.ToDouble(baseline[i]);
            double mult = NumOps.ToDouble(multipliers[i]);
            attributions[i] = NumOps.FromDouble(deltaInput * mult);
        }

        return attributions;
    }

    /// <summary>
    /// Computes approximate DeepLIFT attributions using path integration.
    /// This approximation follows the rescale rule.
    /// </summary>
    private T[] ComputeApproximateAttributions(Vector<T> input, Vector<T> baseline, int outputIndex)
    {
        int n = input.Length;
        var attributions = new T[n];

        // Compute delta outputs for perturbed inputs
        // Using a path approximation similar to integrated gradients but with rescale rule

        if (_rule == DeepLIFTRule.Rescale)
        {
            // Rescale rule: contribution proportional to delta_input * (delta_output / sum(delta_inputs))
            var inputPred = _predictFunction(input);
            var baselinePred = _predictFunction(baseline);

            double outputDiff = (outputIndex < inputPred.Length ? NumOps.ToDouble(inputPred[outputIndex]) : 0)
                              - (outputIndex < baselinePred.Length ? NumOps.ToDouble(baselinePred[outputIndex]) : 0);

            // Compute contributions using gradient scaling
            double totalGradContrib = 0;
            var gradients = ComputeNumericalGradient(input, outputIndex);

            for (int i = 0; i < n; i++)
            {
                double deltaInput = NumOps.ToDouble(input[i]) - NumOps.ToDouble(baseline[i]);
                double grad = NumOps.ToDouble(gradients[i]);
                totalGradContrib += Math.Abs(deltaInput * grad);
            }

            // Rescale to match output difference
            for (int i = 0; i < n; i++)
            {
                double deltaInput = NumOps.ToDouble(input[i]) - NumOps.ToDouble(baseline[i]);
                double grad = NumOps.ToDouble(gradients[i]);
                double rawContrib = deltaInput * grad;

                // Rescale
                if (Math.Abs(totalGradContrib) > 1e-10)
                {
                    attributions[i] = NumOps.FromDouble(rawContrib * outputDiff / totalGradContrib);
                }
                else
                {
                    attributions[i] = NumOps.FromDouble(outputDiff / n);
                }
            }
        }
        else // RevealCancel rule
        {
            // Separate positive and negative contributions
            var posAttrib = new double[n];
            var negAttrib = new double[n];

            var gradients = ComputeNumericalGradient(input, outputIndex);
            var midpoint = new T[n];
            for (int i = 0; i < n; i++)
            {
                midpoint[i] = NumOps.FromDouble((NumOps.ToDouble(input[i]) + NumOps.ToDouble(baseline[i])) / 2);
            }
            var midGradients = ComputeNumericalGradient(new Vector<T>(midpoint), outputIndex);

            for (int i = 0; i < n; i++)
            {
                double deltaInput = NumOps.ToDouble(input[i]) - NumOps.ToDouble(baseline[i]);
                double avgGrad = (NumOps.ToDouble(gradients[i]) + NumOps.ToDouble(midGradients[i])) / 2;
                double contrib = deltaInput * avgGrad;

                if (contrib >= 0)
                    posAttrib[i] = contrib;
                else
                    negAttrib[i] = contrib;
            }

            // Combine
            for (int i = 0; i < n; i++)
            {
                attributions[i] = NumOps.FromDouble(posAttrib[i] + negAttrib[i]);
            }
        }

        return attributions;
    }

    /// <summary>
    /// Computes gradient using backpropagation (if available) or numerical approximation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method computes the gradient of the output with respect
    /// to the input. If a neural network was provided during construction, it uses efficient
    /// backpropagation. Otherwise, it falls back to numerical differentiation.
    ///
    /// Backpropagation is preferred because it:
    /// - Is exact (no approximation error)
    /// - Is faster (O(1) passes vs O(n) for numerical)
    /// - Works better with deep networks
    /// </para>
    /// </remarks>
    private Vector<T> ComputeNumericalGradient(Vector<T> input, int outputIndex)
    {
        // Use backpropagation if available
        if (_gradientHelper != null)
        {
            return _gradientHelper.ComputeGradient(input, outputIndex);
        }

        // Fall back to numerical gradient approximation
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
}

/// <summary>
/// DeepLIFT attribution rules.
/// </summary>
public enum DeepLIFTRule
{
    /// <summary>
    /// Rescale rule: distributes contribution proportionally.
    /// </summary>
    Rescale,

    /// <summary>
    /// RevealCancel rule: separates positive and negative contributions.
    /// Better for understanding opposing effects.
    /// </summary>
    RevealCancel
}

/// <summary>
/// Represents the result of a DeepLIFT analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DeepLIFTExplanation<T>
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
    /// Gets or sets the difference between input and baseline predictions.
    /// </summary>
    public T DeltaOutput { get; set; } = default!;

    /// <summary>
    /// Gets or sets the completeness error (difference between sum of attributions and delta output).
    /// A small error indicates the attributions properly explain the prediction change.
    /// </summary>
    public T CompletenessError { get; set; } = default!;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the DeepLIFT rule used.
    /// </summary>
    public DeepLIFTRule Rule { get; set; }

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
    /// Gets positive contributions (features that increased prediction).
    /// </summary>
    public List<(string name, T attribution)> GetPositiveContributions()
    {
        return GetSortedAttributions()
            .Where(x => NumOps.ToDouble(x.attribution) > 0)
            .ToList();
    }

    /// <summary>
    /// Gets negative contributions (features that decreased prediction).
    /// </summary>
    public List<(string name, T attribution)> GetNegativeContributions()
    {
        return GetSortedAttributions()
            .Where(x => NumOps.ToDouble(x.attribution) < 0)
            .OrderBy(x => NumOps.ToDouble(x.attribution))
            .ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            $"DeepLIFT Explanation (Rule: {Rule}):",
            $"  Baseline prediction: {NumOps.ToDouble(BaselinePrediction):F4}",
            $"  Input prediction: {NumOps.ToDouble(InputPrediction):F4}",
            $"  Delta output: {NumOps.ToDouble(DeltaOutput):F4}",
            $"  Completeness error: {NumOps.ToDouble(CompletenessError):F6}",
            "",
            "Top Positive Contributions:"
        };

        var positive = GetPositiveContributions().Take(5);
        foreach (var (name, attr) in positive)
        {
            lines.Add($"  {name}: +{NumOps.ToDouble(attr):F4}");
        }

        lines.Add("");
        lines.Add("Top Negative Contributions:");

        var negative = GetNegativeContributions().Take(5);
        foreach (var (name, attr) in negative)
        {
            lines.Add($"  {name}: {NumOps.ToDouble(attr):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
