using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes recall (sensitivity, true positive rate): the proportion of actual positives correctly identified.
/// </summary>
/// <remarks>
/// <para>
/// Recall = TP / (TP + FN) = True Positives / Actual Positives
/// </para>
/// <para>
/// <b>For Beginners:</b> Recall answers: "Of all actual positives, how many did the model find?"
/// High recall means few false negatives (missed positives). A cancer screening test with 99% recall
/// means it catches 99% of actual cancer cases.
/// </para>
/// <para>
/// <b>When to prioritize recall:</b>
/// <list type="bullet">
/// <item>When false negatives are costly (missing disease, missing fraud)</item>
/// <item>When you must catch as many positives as possible</item>
/// </list>
/// </para>
/// <para>
/// <b>Trade-off:</b> Improving recall often decreases precision, and vice versa.
/// F1-score balances both, and you can use F-beta to weight one more than the other.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RecallMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _positiveLabel;
    private readonly AveragingMethod _averaging;

    /// <inheritdoc/>
    public string Name => _averaging == AveragingMethod.None ? "Recall" : $"Recall_{_averaging}";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Proportion of actual positives that were correctly identified (TP / (TP + FN)).";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.HigherIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => NumOps.One;

    /// <inheritdoc/>
    public bool RequiresProbabilities => false;

    /// <inheritdoc/>
    public bool SupportsMultiClass => true;

    /// <summary>
    /// Initializes a new recall metric with default positive label (1).
    /// </summary>
    /// <param name="averaging">Averaging method for multi-class.</param>
    public RecallMetric(AveragingMethod averaging = AveragingMethod.Binary)
    {
        _positiveLabel = NumOps.One;
        _averaging = averaging;
    }

    /// <summary>
    /// Initializes a new recall metric with an explicit positive label.
    /// </summary>
    /// <param name="positiveLabel">The label considered positive for binary classification.</param>
    /// <param name="averaging">Averaging method for multi-class.</param>
    public RecallMetric(T positiveLabel, AveragingMethod averaging = AveragingMethod.Binary)
    {
        _positiveLabel = positiveLabel;
        _averaging = averaging;
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
        {
            throw new ArgumentException("Predictions and actuals must have the same length.");
        }

        if (predictions.Length == 0)
        {
            return NumOps.Zero;
        }

        var predictionTensor = EvaluationTensorOps<T>.ToTensor(predictions);
        var actualTensor = EvaluationTensorOps<T>.ToTensor(actuals);

        if (_averaging == AveragingMethod.Binary || _averaging == AveragingMethod.None)
        {
            return ComputeBinaryRecall(predictionTensor, actualTensor, _positiveLabel);
        }

        // Get unique classes from both predictions and actuals
        var classes = EvaluationTensorOps<T>.UniqueLabels(predictionTensor, actualTensor);

        if (_averaging == AveragingMethod.Micro)
        {
            return ComputeMicroRecall(predictionTensor, actualTensor, classes);
        }

        // Macro or Weighted averaging
        double sumRecall = 0;
        double sumWeights = 0;

        foreach (T cls in classes)
        {
            var counts = EvaluationTensorOps<T>.LabelCounts(predictionTensor, actualTensor, cls);
            double recall = RecallFromCounts(counts.TruePositives, counts.ActualPositives);
            int classCount = counts.ActualPositives;

            if (_averaging == AveragingMethod.Macro)
            {
                sumRecall += recall;
                sumWeights += 1;
            }
            else // Weighted
            {
                sumRecall += recall * classCount;
                sumWeights += classCount;
            }
        }

        return sumWeights > 0 ? NumOps.FromDouble(sumRecall / sumWeights) : NumOps.Zero;
    }

    private static T ComputeBinaryRecall(Tensor<T> predictions, Tensor<T> actuals, T positiveLabel)
    {
        var counts = EvaluationTensorOps<T>.LabelCounts(predictions, actuals, positiveLabel);
        return NumOps.FromDouble(RecallFromCounts(counts.TruePositives, counts.ActualPositives));
    }

    private static double RecallFromCounts(int truePositives, int actualPositives)
        => actualPositives > 0 ? (double)truePositives / actualPositives : 0;

    private static T ComputeMicroRecall(Tensor<T> predictions, Tensor<T> actuals, Tensor<T> classes)
    {
        int totalTruePositives = 0;
        int totalActualPositives = 0;

        foreach (T cls in classes)
        {
            var counts = EvaluationTensorOps<T>.LabelCounts(predictions, actuals, cls);
            totalActualPositives += counts.ActualPositives;
            totalTruePositives += counts.TruePositives;
        }

        return totalActualPositives > 0 ? NumOps.FromDouble((double)totalTruePositives / totalActualPositives) : NumOps.Zero;
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = ComputeBootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);

        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T lower, T upper) ComputeBootstrapCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        int bootstrapSamples,
        double confidenceLevel,
        int? randomSeed)
    {
        int n = predictions.Length;
        if (n == 0)
        {
            return (NumOps.Zero, NumOps.One);
        }

        var random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        var bootstrapValues = new List<T>(bootstrapSamples);
        var predArray = predictions.ToArray();
        var actualArray = actuals.ToArray();

        for (int b = 0; b < bootstrapSamples; b++)
        {
            var sampledPred = new T[n];
            var sampledActual = new T[n];

            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sampledPred[i] = predArray[idx];
                sampledActual[i] = actualArray[idx];
            }

            bootstrapValues.Add(Compute(sampledPred, sampledActual));
        }

        double alpha = 1 - confidenceLevel;
        int lowerIdx = (int)Math.Floor(alpha / 2 * bootstrapSamples);
        int upperIdx = (int)Math.Ceiling((1 - alpha / 2) * bootstrapSamples) - 1;

        lowerIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, lowerIdx));
        upperIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, upperIdx));

        return EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, lowerIdx, upperIdx);
    }
}
