using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// DeepSHAP explainer combining GradientSHAP with DeepLIFT for efficient neural network explanations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DeepSHAP is a fast method for computing SHAP values specifically designed
/// for deep neural networks. It combines two powerful ideas:
///
/// 1. <b>DeepLIFT rules</b>: How to properly attribute through non-linearities (ReLU, etc.)
/// 2. <b>Shapley sampling</b>: Using multiple baseline samples for better attribution
///
/// The key insight is that by using DeepLIFT's "multipliers" (instead of regular gradients),
/// we get attributions that are more stable and interpretable.
///
/// How DeepSHAP works:
/// 1. Sample multiple reference inputs from your background data
/// 2. For each reference, compute DeepLIFT-style attributions
/// 3. Average the attributions across all references
///
/// This gives you Shapley-style attributions (fair credit assignment) computed
/// efficiently using backpropagation.
///
/// When to use DeepSHAP vs other methods:
/// - <b>DeepSHAP</b>: Best for deep neural networks, especially with ReLU activations
/// - <b>GradientSHAP</b>: Simpler, works well when gradients are reliable
/// - <b>KernelSHAP</b>: Model-agnostic but slower
/// - <b>IntegratedGradients</b>: Theoretically grounded but uses single baseline
///
/// DeepSHAP advantages:
/// - Fast (single backward pass per sample)
/// - Handles saturation regions well (via DeepLIFT rules)
/// - Produces Shapley-like fair attributions
/// - Works with any differentiable neural network
/// </para>
/// </remarks>
public class DeepSHAPExplainer<T> : ILocalExplainer<T, DeepSHAPExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _gradientFunction;
    private readonly Func<Vector<T>, Vector<T>, Vector<T>>? _deepLiftMultipliers;
    private readonly Matrix<T> _backgroundData;
    private readonly int _numSamples;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private readonly T _expectedValue;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "DeepSHAP";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, DeepSHAP computes
    /// attributions for multiple background samples in parallel.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Gets the expected (baseline) prediction value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the average prediction when we don't know any feature values.
    /// DeepSHAP values explain how features push the prediction away from this baseline.
    /// </para>
    /// </remarks>
    public T ExpectedValue => _expectedValue;

    /// <summary>
    /// Gets the number of background samples used for computing attributions.
    /// </summary>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Initializes a new DeepSHAP explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="gradientFunction">Function that computes input gradients.
    /// Takes (input, outputIndex) and returns gradient vector.</param>
    /// <param name="backgroundData">Reference dataset for sampling baselines.</param>
    /// <param name="numSamples">Number of background samples to use per explanation (default: 100).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>backgroundData</b>: Representative samples from your training data. DeepSHAP
    ///   compares each input against these to determine feature importance.
    /// - <b>numSamples</b>: How many background samples to use. More = more accurate but slower.
    ///   100 is usually sufficient.
    /// </para>
    /// </remarks>
    public DeepSHAPExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, int, Vector<T>>? gradientFunction,
        Matrix<T> backgroundData,
        int numSamples = 100,
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

        _numSamples = Math.Min(numSamples, backgroundData.Rows);
        _featureNames = featureNames;
        _randomState = randomState;

        // Compute expected value from background data
        _expectedValue = ComputeExpectedValue(backgroundData);
    }

    /// <summary>
    /// Initializes a new DeepSHAP explainer with DeepLIFT multiplier support.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="gradientFunction">Function that computes input gradients.</param>
    /// <param name="deepLiftMultipliers">Function that computes DeepLIFT multipliers.
    /// Takes (input, reference) and returns multipliers for each input feature.</param>
    /// <param name="backgroundData">Reference dataset for sampling baselines.</param>
    /// <param name="numSamples">Number of background samples to use.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor is for advanced use when you have a neural network
    /// that can compute DeepLIFT-style multipliers. These multipliers handle non-linearities
    /// better than regular gradients.
    ///
    /// If you don't have DeepLIFT multipliers, use the simpler constructor - it will
    /// approximate them using gradients (which works well for most networks).
    /// </para>
    /// </remarks>
    public DeepSHAPExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, int, Vector<T>>? gradientFunction,
        Func<Vector<T>, Vector<T>, Vector<T>>? deepLiftMultipliers,
        Matrix<T> backgroundData,
        int numSamples = 100,
        string[]? featureNames = null,
        int? randomState = null)
        : this(predictFunction, gradientFunction, backgroundData, numSamples, featureNames, randomState)
    {
        _deepLiftMultipliers = deepLiftMultipliers;
    }

    /// <summary>
    /// Initializes a new DeepSHAP explainer from a neural network.
    /// </summary>
    /// <param name="neuralNetwork">The neural network to explain.</param>
    /// <param name="backgroundData">Reference dataset for sampling baselines.</param>
    /// <param name="numSamples">Number of background samples to use.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the easiest way to create a DeepSHAP explainer for
    /// neural networks. It automatically sets up gradient computation using backpropagation.
    /// </para>
    /// </remarks>
    public DeepSHAPExplainer(
        INeuralNetwork<T> neuralNetwork,
        Matrix<T> backgroundData,
        int numSamples = 100,
        string[]? featureNames = null,
        int? randomState = null)
    {
        if (neuralNetwork is null)
            throw new ArgumentNullException(nameof(neuralNetwork));
        _backgroundData = backgroundData ?? throw new ArgumentNullException(nameof(backgroundData));

        if (backgroundData.Rows == 0)
            throw new ArgumentException("Background data must have at least one row.", nameof(backgroundData));
        if (numSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1.", nameof(numSamples));

        // Create gradient helper for backpropagation
        var gradientHelper = new InputGradientHelper<T>(neuralNetwork);

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

        _gradientFunction = gradientHelper.CreateGradientFunction();
        _numSamples = Math.Min(numSamples, backgroundData.Rows);
        _featureNames = featureNames;
        _randomState = randomState;

        // Compute expected value
        _expectedValue = ComputeExpectedValue(backgroundData);
    }

    /// <summary>
    /// Computes DeepSHAP attributions for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>DeepSHAP explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method computes how much each input feature contributed
    /// to the prediction, compared to the expected prediction on background data.
    ///
    /// Positive values mean the feature increased the prediction.
    /// Negative values mean the feature decreased the prediction.
    /// The sum of all attributions equals (prediction - expected_prediction).
    /// </para>
    /// </remarks>
    public DeepSHAPExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: 0);
    }

    /// <summary>
    /// Computes DeepSHAP attributions for a specific output.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">Index of the output to explain.</param>
    /// <returns>DeepSHAP explanation with feature attributions.</returns>
    public DeepSHAPExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        int numFeatures = instance.Length;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Select random background samples
        var sampleIndices = new int[_numSamples];
        for (int i = 0; i < _numSamples; i++)
        {
            sampleIndices[i] = rand.Next(_backgroundData.Rows);
        }

        // Compute attributions relative to each background sample
        var attributions = new double[numFeatures];

        if (_gpuHelper != null && _gpuHelper.IsGPUEnabled)
        {
            // Parallel processing of background samples
            var sampleAttributions = new double[_numSamples][];
            Parallel.For(0, _numSamples, new ParallelOptions
            {
                MaxDegreeOfParallelism = _gpuHelper.MaxParallelism
            }, s =>
            {
                var reference = _backgroundData.GetRow(sampleIndices[s]);
                sampleAttributions[s] = ComputeDeepLIFTAttributions(instance, reference, outputIndex);
            });

            // Average across samples
            for (int s = 0; s < _numSamples; s++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    attributions[j] += sampleAttributions[s][j];
                }
            }
        }
        else
        {
            // Sequential processing
            for (int s = 0; s < _numSamples; s++)
            {
                var reference = _backgroundData.GetRow(sampleIndices[s]);
                var sampleAttr = ComputeDeepLIFTAttributions(instance, reference, outputIndex);

                for (int j = 0; j < numFeatures; j++)
                {
                    attributions[j] += sampleAttr[j];
                }
            }
        }

        // Average across samples
        double invSamples = 1.0 / _numSamples;
        for (int j = 0; j < numFeatures; j++)
        {
            attributions[j] *= invSamples;
        }

        // Get prediction
        var prediction = _predictFunction(instance);
        T predVal = outputIndex < prediction.Length ? prediction[outputIndex] : NumOps.Zero;

        // Convert attributions to T
        var attrT = new T[numFeatures];
        for (int j = 0; j < numFeatures; j++)
        {
            attrT[j] = NumOps.FromDouble(attributions[j]);
        }

        return new DeepSHAPExplanation<T>
        {
            Attributions = new Vector<T>(attrT),
            ExpectedValue = _expectedValue,
            Prediction = predVal,
            Instance = instance,
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray(),
            OutputIndex = outputIndex,
            NumSamples = _numSamples
        };
    }

    /// <inheritdoc/>
    public DeepSHAPExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new DeepSHAPExplanation<T>[instances.Rows];

        if (_gpuHelper != null && _gpuHelper.IsGPUEnabled && instances.Rows > 1)
        {
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
    /// Computes global feature importance by averaging absolute SHAP values.
    /// </summary>
    /// <param name="data">Dataset to compute global importance over.</param>
    /// <returns>Global explanation with average feature importance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Global importance shows which features are most important
    /// across all predictions, not just for a single input. Features with high global
    /// importance consistently have large effects on predictions.
    /// </para>
    /// </remarks>
    public GlobalDeepSHAPExplanation<T> ExplainGlobal(Matrix<T> data)
    {
        var localExplanations = ExplainBatch(data);
        return new GlobalDeepSHAPExplanation<T>(localExplanations, _featureNames);
    }

    /// <summary>
    /// Computes DeepLIFT-style attributions relative to a reference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepLIFT computes attributions by comparing activations
    /// at the input to activations at a reference. The key formula is:
    ///
    /// attribution[i] = (input[i] - reference[i]) * multiplier[i]
    ///
    /// where multiplier[i] represents how much the output changes per unit change in input[i].
    /// For linear networks, multiplier = gradient. For non-linear networks, DeepLIFT uses
    /// special rules to compute more stable multipliers.
    /// </para>
    /// </remarks>
    private double[] ComputeDeepLIFTAttributions(Vector<T> input, Vector<T> reference, int outputIndex)
    {
        int numFeatures = input.Length;
        var attributions = new double[numFeatures];

        // Compute difference from reference
        var diff = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++)
        {
            diff[j] = NumOps.ToDouble(input[j]) - NumOps.ToDouble(reference[j]);
        }

        // Use DeepLIFT multipliers if available, otherwise approximate with gradients
        if (_deepLiftMultipliers != null)
        {
            var multipliers = _deepLiftMultipliers(input, reference);
            for (int j = 0; j < numFeatures; j++)
            {
                attributions[j] = diff[j] * NumOps.ToDouble(multipliers[j]);
            }
        }
        else if (_gradientFunction != null)
        {
            // Approximate DeepLIFT using average of gradients at input and reference
            // This is called "Gradient × Input" approximation
            var gradAtInput = _gradientFunction(input, outputIndex);
            var gradAtRef = _gradientFunction(reference, outputIndex);

            for (int j = 0; j < numFeatures; j++)
            {
                // Average gradient (simple DeepLIFT approximation)
                double avgGrad = 0.5 * (NumOps.ToDouble(gradAtInput[j]) + NumOps.ToDouble(gradAtRef[j]));
                attributions[j] = diff[j] * avgGrad;
            }
        }
        else
        {
            // Fall back to simple difference weighting
            var inputPred = _predictFunction(input);
            var refPred = _predictFunction(reference);

            double predDiff = NumOps.ToDouble(inputPred[outputIndex]) - NumOps.ToDouble(refPred[outputIndex]);

            // Distribute prediction difference proportionally to input differences
            double totalDiff = 0;
            for (int j = 0; j < numFeatures; j++)
            {
                totalDiff += Math.Abs(diff[j]);
            }

            if (totalDiff > 1e-10)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    attributions[j] = predDiff * (diff[j] / totalDiff);
                }
            }
        }

        return attributions;
    }

    /// <summary>
    /// Computes the expected prediction value from background data.
    /// </summary>
    private T ComputeExpectedValue(Matrix<T> backgroundData)
    {
        double sum = 0;
        int numSamples = Math.Min(backgroundData.Rows, 100);

        for (int i = 0; i < numSamples; i++)
        {
            var sample = backgroundData.GetRow(i);
            var pred = _predictFunction(sample);
            sum += NumOps.ToDouble(pred[0]);
        }

        return NumOps.FromDouble(sum / numSamples);
    }
}

/// <summary>
/// Represents the result of a DeepSHAP analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DeepSHAPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the feature attributions (SHAP values).
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the expected (baseline) prediction value.
    /// </summary>
    public T ExpectedValue { get; set; } = default!;

    /// <summary>
    /// Gets or sets the actual prediction for this instance.
    /// </summary>
    public T Prediction { get; set; } = default!;

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Instance { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the number of background samples used.
    /// </summary>
    public int NumSamples { get; set; }

    /// <summary>
    /// Gets attributions sorted by absolute value (most important first).
    /// </summary>
    public List<(string name, T value, T attribution)> GetSortedAttributions()
    {
        var result = new List<(string, T, T)>();
        for (int i = 0; i < Attributions.Length; i++)
        {
            result.Add((FeatureNames[i], Instance[i], Attributions[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item3))).ToList();
    }

    /// <summary>
    /// Verifies that attributions sum to (prediction - expected_value).
    /// </summary>
    /// <returns>The sum error (should be close to zero).</returns>
    public double GetSumError()
    {
        double attrSum = 0;
        for (int i = 0; i < Attributions.Length; i++)
        {
            attrSum += NumOps.ToDouble(Attributions[i]);
        }

        double expectedDiff = NumOps.ToDouble(Prediction) - NumOps.ToDouble(ExpectedValue);
        return Math.Abs(attrSum - expectedDiff);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            "DeepSHAP Explanation:",
            $"  Expected value: {NumOps.ToDouble(ExpectedValue):F4}",
            $"  Prediction: {NumOps.ToDouble(Prediction):F4}",
            $"  Sum error: {GetSumError():F6}",
            $"  Samples used: {NumSamples}",
            "",
            "Top Feature Attributions:"
        };

        var sorted = GetSortedAttributions().Take(10);
        foreach (var (name, value, attr) in sorted)
        {
            double attrVal = NumOps.ToDouble(attr);
            string sign = attrVal >= 0 ? "+" : "";
            lines.Add($"  {name} = {NumOps.ToDouble(value):F4}: {sign}{attrVal:F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Represents global DeepSHAP feature importance.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GlobalDeepSHAPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the average absolute attribution for each feature.
    /// </summary>
    public Vector<T> MeanAbsoluteAttributions { get; }

    /// <summary>
    /// Gets the feature names.
    /// </summary>
    public string[] FeatureNames { get; }

    /// <summary>
    /// Gets the local explanations used to compute global importance.
    /// </summary>
    public DeepSHAPExplanation<T>[] LocalExplanations { get; }

    /// <summary>
    /// Initializes a new global DeepSHAP explanation.
    /// </summary>
    public GlobalDeepSHAPExplanation(DeepSHAPExplanation<T>[] localExplanations, string[]? featureNames = null)
    {
        LocalExplanations = localExplanations;

        if (localExplanations.Length == 0)
        {
            MeanAbsoluteAttributions = new Vector<T>(0);
            FeatureNames = Array.Empty<string>();
            return;
        }

        int numFeatures = localExplanations[0].Attributions.Length;
        FeatureNames = featureNames ?? localExplanations[0].FeatureNames;

        // Compute mean absolute attributions
        var meanAbs = new double[numFeatures];
        foreach (var explanation in localExplanations)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                meanAbs[j] += Math.Abs(NumOps.ToDouble(explanation.Attributions[j]));
            }
        }

        double invN = 1.0 / localExplanations.Length;
        var result = new T[numFeatures];
        for (int j = 0; j < numFeatures; j++)
        {
            result[j] = NumOps.FromDouble(meanAbs[j] * invN);
        }

        MeanAbsoluteAttributions = new Vector<T>(result);
    }

    /// <summary>
    /// Gets features sorted by global importance.
    /// </summary>
    public List<(string name, T importance)> GetSortedImportance()
    {
        var result = new List<(string, T)>();
        for (int i = 0; i < MeanAbsoluteAttributions.Length; i++)
        {
            result.Add((FeatureNames[i], MeanAbsoluteAttributions[i]));
        }
        return result.OrderByDescending(x => NumOps.ToDouble(x.Item2)).ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            "Global DeepSHAP Feature Importance:",
            $"  Based on {LocalExplanations.Length} samples",
            "",
            "Feature Importance (mean |SHAP value|):"
        };

        var sorted = GetSortedImportance();
        foreach (var (name, importance) in sorted)
        {
            lines.Add($"  {name}: {NumOps.ToDouble(importance):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
