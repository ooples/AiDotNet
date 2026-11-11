using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Feature Ablation explainer for understanding feature importance by removal.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Feature Ablation is a simple way to understand feature importance.
/// The idea is: replace each feature with a "baseline" value and see how the prediction changes.
///
/// <b>How it works:</b>
/// 1. For each feature (or group of features):
/// 2. Replace the feature with a baseline (e.g., zero, mean, or reference value)
/// 3. Measure how much the prediction changes
/// 4. Features that cause large changes are important
///
/// <b>Difference from Occlusion:</b>
/// - Occlusion: Slides a window over spatial data (images)
/// - Feature Ablation: Works on individual features (tabular data, any modality)
///
/// <b>Key advantage:</b> You can group features and ablate them together:
/// - All color channels for an image
/// - All words in a sentence segment
/// - All related features in tabular data
///
/// <b>Use cases:</b>
/// - Understanding which features drive predictions in tabular data
/// - Grouping related features (e.g., all "income" related columns)
/// - Debugging models by finding unexpected important features
/// </para>
/// </remarks>
public partial class FeatureAblationExplainer<T> : ILocalExplainer<T, FeatureAblationExplanation<T>>, IGlobalExplainer<T, GlobalFeatureAblationResult<T>>, IGlobalAttributionExplainer<T>, IGPUAcceleratedExplainer<T>
{
    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Tensor<T>, Tensor<T>>? _tensorPredictFunction;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T>? _baseline;
    private readonly int[][]? _featureGroups;
    private readonly string[]? _featureNames;
    private readonly bool _perturbEachSeparately;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <summary>
    /// Gets the method name.
    /// </summary>
    public string MethodName => "FeatureAblation";

    /// <summary>
    /// Gets whether this explainer supports local explanations.
    /// </summary>
    public bool SupportsLocalExplanations => true;

    /// <summary>
    /// Gets whether this explainer supports global explanations.
    /// </summary>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new Feature Ablation explainer.
    /// </summary>
    /// <param name="predictFunction">Function that takes input vector and returns predictions.</param>
    /// <param name="baseline">Baseline values to use when ablating (default: zeros).</param>
    /// <param name="featureGroups">Groups of feature indices to ablate together (default: each feature separately).</param>
    /// <param name="featureNames">Names for features or groups.</param>
    /// <param name="perturbEachSeparately">If true, ablate each feature independently. If false, use cumulative ablation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>baseline:</b> What to replace features with. Common choices:
    ///   - Zero: Simple, works for normalized data
    ///   - Mean: Average feature value from training data
    ///   - Median: More robust to outliers
    /// - <b>featureGroups:</b> Which features to ablate together. Examples:
    ///   - [[0,1,2], [3,4,5]] = First group is features 0,1,2; second is 3,4,5
    ///   - null = Each feature is its own group
    /// - <b>perturbEachSeparately:</b>
    ///   - true = Independent ablation (recommended for importance)
    ///   - false = Cumulative (shows compounding effects)
    /// </para>
    /// </remarks>
    public FeatureAblationExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Vector<T>? baseline = null,
        int[][]? featureGroups = null,
        string[]? featureNames = null,
        bool perturbEachSeparately = true)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        _baseline = baseline;
        _featureGroups = featureGroups;
        _featureNames = featureNames;
        _perturbEachSeparately = perturbEachSeparately;
    }

    /// <summary>
    /// Initializes a Feature Ablation explainer for tensor inputs.
    /// </summary>
    /// <param name="tensorPredictFunction">Function that takes input tensor and returns predictions.</param>
    /// <param name="featureGroups">Groups of feature indices to ablate together.</param>
    /// <param name="featureNames">Names for features or groups.</param>
    /// <param name="perturbEachSeparately">If true, ablate each feature independently.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor for image or tensor inputs.
    /// For images, you might group features like:
    /// - All pixels in a region
    /// - All channels at a position
    /// - Semantic segments
    /// </para>
    /// </remarks>
    public FeatureAblationExplainer(
        Func<Tensor<T>, Tensor<T>> tensorPredictFunction,
        int[][]? featureGroups = null,
        string[]? featureNames = null,
        bool perturbEachSeparately = true)
    {
        Guard.NotNull(tensorPredictFunction);
        _tensorPredictFunction = tensorPredictFunction;
        _featureGroups = featureGroups;
        _featureNames = featureNames;
        _perturbEachSeparately = perturbEachSeparately;

        _predictFunction = input =>
        {
            return tensorPredictFunction(ExplainerTensorOps<T>.RowTensor(input)).ToVector();
        };
    }

    /// <summary>
    /// Explains a single input by ablating features.
    /// </summary>
    /// <param name="input">The input vector to explain.</param>
    /// <param name="targetClass">Target class to explain (default: predicted class).</param>
    /// <returns>Feature ablation explanation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns how much each feature (or group) contributes
    /// to the prediction. Positive attribution = feature increases prediction for target class.
    /// </para>
    /// </remarks>
    public FeatureAblationExplanation<T> ExplainLocal(Vector<T> input, int? targetClass = null)
    {
        // Determine baseline
        var baseline = _baseline ?? new Vector<T>(input.Length);

        // Get base prediction
        var basePrediction = _predictFunction(input);
        int actualTarget = targetClass ?? GetPredictedClass(basePrediction);
        T baseScore = basePrediction[actualTarget];

        // Determine feature groups
        var groups = _featureGroups ?? CreateDefaultGroups(input.Length);
        int numGroups = groups.Length;

        // Compute ablated predictions for each group
        var ablatedPredictions = new List<Vector<T>>(numGroups);
        var cumulativeAblated = input.Clone();
        for (int g = 0; g < numGroups; g++)
        {
            if (_perturbEachSeparately)
            {
                // Independent ablation
                ablatedPredictions.Add(_predictFunction(AblateFeatures(input, baseline, groups[g])));
            }
            else
            {
                // Cumulative ablation (Shapley-like order)
                cumulativeAblated = AblateFeatures(cumulativeAblated, baseline, groups[g]);
                ablatedPredictions.Add(_predictFunction(cumulativeAblated));
            }
        }

        var engine = AiDotNetEngine.Current;
        var groupScores = new Vector<T>(numGroups);
        var attributions = new Vector<T>(numGroups);
        if (numGroups > 0)
        {
            var scores = engine.TensorSliceAxis(ExplainerTensorOps<T>.Stack(ablatedPredictions), 1, actualTarget);
            groupScores = scores.ToVector();

            // Independent: attribution = drop in score when the feature is removed.
            // Cumulative: attribution = marginal contribution, the drop from the previous step's score.
            var previousScores = _perturbEachSeparately || numGroups == 1
                ? engine.TensorFullLike(scores, baseScore)
                : engine.TensorConcatenate(new[]
                {
                    engine.TensorFullLike(new Tensor<T>(new[] { 1 }), baseScore),
                    engine.TensorSlice(scores, new[] { 0 }, new[] { numGroups - 1 })
                }, 0);
            attributions = engine.TensorSubtract(previousScores, scores).ToVector();
        }

        // Get feature/group names
        var names = _featureNames ?? groups.Select((g, i) =>
            g.Length == 1 ? $"Feature_{g[0]}" : $"Group_{i}").ToArray();

        return new FeatureAblationExplanation<T>(
            input: input,
            attributions: attributions,
            baseline: baseline,
            featureGroups: groups,
            featureNames: names,
            targetClass: actualTarget,
            basePrediction: baseScore,
            ablatedPredictions: groupScores);
    }

    /// <summary>
    /// Computes global feature importance across a dataset.
    /// </summary>
    /// <param name="data">Dataset to analyze (rows = samples).</param>
    /// <returns>Global feature ablation result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives you the average importance of each feature
    /// across many samples. More reliable than single-sample explanations.
    /// </para>
    /// </remarks>
    public GlobalFeatureAblationResult<T> ExplainGlobal(Matrix<T> data)
    {
        return ExplainGlobal(data, targetClass: null);
    }

    /// <summary>
    /// Computes global feature importance across a dataset for a specific class.
    /// </summary>
    /// <param name="data">Dataset to analyze (rows = samples).</param>
    /// <param name="targetClass">Target class to explain.</param>
    /// <returns>Global feature ablation result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives you the average importance of each feature
    /// across many samples for a specific target class.
    /// </para>
    /// </remarks>
    public GlobalFeatureAblationResult<T> ExplainGlobal(Matrix<T> data, int? targetClass)
    {
        int numSamples = data.Rows;
        int numFeatures = data.Columns;

        var groups = _featureGroups ?? CreateDefaultGroups(numFeatures);
        int numGroups = groups.Length;

        // Collect attributions
        var sampleAttributions = new List<Vector<T>>(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            var explanation = ExplainLocal(data.GetRow(i), targetClass);
            sampleAttributions.Add(explanation.Attributions.GetSubVector(0, numGroups));
        }

        // Compute statistics: mean, mean |attribution| and population std per group
        var meanAttributions = new Vector<T>(numGroups);
        var meanAbsAttributions = new Vector<T>(numGroups);
        var stdAttributions = new Vector<T>(numGroups);
        if (numSamples > 0 && numGroups > 0)
        {
            var engine = AiDotNetEngine.Current;
            var stacked = ExplainerTensorOps<T>.Stack(sampleAttributions);
            var mean = engine.ReduceMean(stacked, new[] { 0 }, true);
            var deviations = engine.TensorSubtract(stacked, engine.TensorBroadcastTo(mean, stacked.Shape));
            meanAttributions = mean.ToVector();
            meanAbsAttributions = engine.ReduceMean(engine.TensorAbs(stacked), new[] { 0 }, false).ToVector();
            stdAttributions = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(deviations), new[] { 0 }, false)).ToVector();
        }

        var names = _featureNames ?? groups.Select((g, i) =>
            g.Length == 1 ? $"Feature_{g[0]}" : $"Group_{i}").ToArray();

        return new GlobalFeatureAblationResult<T>(
            featureGroups: groups,
            featureNames: names,
            meanAttributions: meanAttributions,
            meanAbsoluteAttributions: meanAbsAttributions,
            stdAttributions: stdAttributions,
            numSamples: numSamples);
    }

    /// <summary>
    /// Ablates specified features in the input.
    /// </summary>
    private Vector<T> AblateFeatures(Vector<T> input, Vector<T> baseline, int[] featureIndices)
    {
        // Replace the group's (in-range) features by the baseline, or zero past the baseline's end
        var ablate = new bool[input.Length];
        foreach (var idx in featureIndices)
        {
            if (idx >= 0 && idx < ablate.Length)
            {
                ablate[idx] = true;
            }
        }

        var replacement = ExplainerTensorOps<T>.PadOrTruncate(baseline, input.Length);
        return AiDotNetEngine.Current.TensorWhere(ablate, replacement.AsTensor(), input.AsTensor()).ToVector();
    }

    /// <summary>
    /// Creates default feature groups (one per feature).
    /// </summary>
    private int[][] CreateDefaultGroups(int numFeatures)
    {
        return Enumerable.Range(0, numFeatures)
            .Select(i => new[] { i })
            .ToArray();
    }

    /// <summary>
    /// Gets the predicted class from output.
    /// </summary>
    private int GetPredictedClass(Vector<T> output)
    {
        return ExplainerTensorOps<T>.ArgMax(output);
    }

    /// <summary>
    /// Explains a single input (ILocalExplainer interface).
    /// </summary>
    public FeatureAblationExplanation<T> Explain(Vector<T> instance)
    {
        return ExplainLocal(instance);
    }

    /// <summary>
    /// Explains a batch of inputs (ILocalExplainer interface).
    /// </summary>
    /// <inheritdoc />
    public Vector<T> ComputeGlobalAttributions(Matrix<T> data)
        => AttributionAggregation.MeanAbsolute<T>(ExplainBatch(data));

    public FeatureAblationExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var results = new FeatureAblationExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            results[i] = ExplainLocal(instances.GetRow(i));
        }
        return results;
    }
}

/// <summary>
/// Result of feature ablation for a single input.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FeatureAblationExplanation<T> : AiDotNet.Interpretability.IFeatureAttribution<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the original input.
    /// </summary>
    public Vector<T> Input { get; }

    /// <summary>
    /// Gets attributions for each feature/group.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Positive = feature helps the prediction.
    /// Negative = feature hurts the prediction.
    /// Larger magnitude = more important.
    /// </para>
    /// </remarks>
    public Vector<T> Attributions { get; }

    /// <inheritdoc />
    public Vector<T> GetFeatureAttributions() => Attributions;

    /// <summary>
    /// Gets the baseline values used.
    /// </summary>
    public Vector<T> Baseline { get; }

    /// <summary>
    /// Gets the feature groups.
    /// </summary>
    public int[][] FeatureGroups { get; }

    /// <summary>
    /// Gets feature/group names.
    /// </summary>
    public string[] FeatureNames { get; }

    /// <summary>
    /// Gets the target class explained.
    /// </summary>
    public int TargetClass { get; }

    /// <summary>
    /// Gets the base prediction score.
    /// </summary>
    public T BasePrediction { get; }

    /// <summary>
    /// Gets predictions after ablating each feature/group.
    /// </summary>
    public Vector<T> AblatedPredictions { get; }

    /// <summary>
    /// Gets the number of feature groups.
    /// </summary>
    public int NumGroups => Attributions.Length;

    /// <summary>
    /// Initializes a new feature ablation explanation.
    /// </summary>
    public FeatureAblationExplanation(
        Vector<T> input,
        Vector<T> attributions,
        Vector<T> baseline,
        int[][] featureGroups,
        string[] featureNames,
        int targetClass,
        T basePrediction,
        Vector<T> ablatedPredictions)
    {
        Input = input;
        Attributions = attributions;
        Baseline = baseline;
        FeatureGroups = featureGroups;
        FeatureNames = featureNames;
        TargetClass = targetClass;
        BasePrediction = basePrediction;
        AblatedPredictions = ablatedPredictions;
    }

    /// <summary>
    /// Gets features sorted by importance (absolute attribution).
    /// </summary>
    public IEnumerable<(string Name, int[] Group, T Attribution)> GetSortedFeatures()
    {
        return Enumerable.Range(0, NumGroups)
            .Select(i => (Name: FeatureNames[i], Group: FeatureGroups[i], Attribution: Attributions[i]))
            .OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Attribution)));
    }

    /// <summary>
    /// Gets the top K most important features.
    /// </summary>
    public IEnumerable<(string Name, int[] Group, T Attribution)> GetTopFeatures(int k = 10)
    {
        return GetSortedFeatures().Take(k);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopFeatures(5).ToList();

        return $"Feature Ablation for class {TargetClass}:\n" +
               $"  Base prediction: {BasePrediction}\n" +
               $"  Top features:\n" +
               string.Join("\n", top.Select(f =>
                   $"    {f.Name}: {f.Attribution:F4}"));
    }
}

/// <summary>
/// Global feature ablation result across a dataset.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GlobalFeatureAblationResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the feature groups.
    /// </summary>
    public int[][] FeatureGroups { get; }

    /// <summary>
    /// Gets feature/group names.
    /// </summary>
    public string[] FeatureNames { get; }

    /// <summary>
    /// Gets mean attributions (can be positive or negative).
    /// </summary>
    public Vector<T> MeanAttributions { get; }

    /// <summary>
    /// Gets mean absolute attributions (importance magnitude).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the most useful measure of feature importance.
    /// It tells you how much each feature affects predictions on average,
    /// regardless of direction.
    /// </para>
    /// </remarks>
    public Vector<T> MeanAbsoluteAttributions { get; }

    /// <summary>
    /// Gets standard deviation of attributions.
    /// </summary>
    public Vector<T> StdAttributions { get; }

    /// <summary>
    /// Gets the number of samples analyzed.
    /// </summary>
    public int NumSamples { get; }

    /// <summary>
    /// Initializes a new global feature ablation result.
    /// </summary>
    public GlobalFeatureAblationResult(
        int[][] featureGroups,
        string[] featureNames,
        Vector<T> meanAttributions,
        Vector<T> meanAbsoluteAttributions,
        Vector<T> stdAttributions,
        int numSamples)
    {
        FeatureGroups = featureGroups;
        FeatureNames = featureNames;
        MeanAttributions = meanAttributions;
        MeanAbsoluteAttributions = meanAbsoluteAttributions;
        StdAttributions = stdAttributions;
        NumSamples = numSamples;
    }

    /// <summary>
    /// Gets features sorted by importance.
    /// </summary>
    public IEnumerable<(string Name, T MeanAbsAttribution, T MeanAttribution, T Std)> GetSortedFeatures()
    {
        return Enumerable.Range(0, FeatureNames.Length)
            .Select(i => (
                Name: FeatureNames[i],
                MeanAbsAttribution: MeanAbsoluteAttributions[i],
                MeanAttribution: MeanAttributions[i],
                Std: StdAttributions[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.MeanAbsAttribution));
    }

    /// <summary>
    /// Gets the top K most important features globally.
    /// </summary>
    public IEnumerable<(string Name, T MeanAbsAttribution, T MeanAttribution, T Std)> GetTopFeatures(int k = 10)
    {
        return GetSortedFeatures().Take(k);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopFeatures(10).ToList();

        return $"Global Feature Ablation ({NumSamples} samples):\n" +
               string.Join("\n", top.Select(f =>
                   $"  {f.Name}: |attr|={f.MeanAbsAttribution:F4} " +
                   $"(mean={f.MeanAttribution:F4}, std={f.Std:F4})"));
    }
}
