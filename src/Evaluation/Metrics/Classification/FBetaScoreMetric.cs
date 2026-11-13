using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes F-beta score: weighted harmonic mean of precision and recall.
/// </summary>
/// <remarks>
/// <para>
/// F_beta = (1 + beta²) * (precision * recall) / (beta² * precision + recall)
/// </para>
/// <para>
/// <b>For Beginners:</b> F-beta lets you weight precision vs recall:
/// <list type="bullet">
/// <item>beta = 1: F1 score (equal weight)</item>
/// <item>beta = 2: F2 score (recall twice as important as precision)</item>
/// <item>beta = 0.5: F0.5 score (precision twice as important as recall)</item>
/// </list>
/// </para>
/// </remarks>
public class FBetaScoreMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly T _positiveLabel;
    private readonly double _beta;
    private readonly AveragingMethod _averaging;

    /// <inheritdoc />
    public string Name => $"F{_beta:F1}Score";
    /// <inheritdoc />
    public string Category => "Classification";
    /// <inheritdoc />
    public string Description => $"F-beta score with beta={_beta}, weighting recall {(_beta > 1 ? "more" : _beta < 1 ? "less" : "equally")} than precision.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => true;

    public FBetaScoreMetric(double beta = 1.0, AveragingMethod averaging = AveragingMethod.Binary)
    {
        if (double.IsNaN(beta) || double.IsInfinity(beta) || beta <= 0)
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be a finite positive value.");
        _beta = beta;
        _positiveLabel = NumOps.One;
        _averaging = averaging;
    }

    public FBetaScoreMetric(double beta, T positiveLabel, AveragingMethod averaging = AveragingMethod.Binary)
    {
        if (double.IsNaN(beta) || double.IsInfinity(beta) || beta <= 0)
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be a finite positive value.");
        _beta = beta;
        _positiveLabel = positiveLabel;
        _averaging = averaging;
    }

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);

        if (_averaging == AveragingMethod.Binary || _averaging == AveragingMethod.None)
            return ComputeBinary(pred, actual, _positiveLabel);

        var classes = EvaluationTensorOps<T>.UniqueLabels(pred, actual);
        if (_averaging == AveragingMethod.Micro)
            return ComputeMicro(pred, actual, classes);

        double sum = 0, weights = 0;
        foreach (T cls in classes)
        {
            var counts = EvaluationTensorOps<T>.LabelCounts(pred, actual, cls);
            double fb = BinaryFromCounts(counts.TruePositives, counts.PredictedPositives, counts.ActualPositives);
            int count = counts.ActualPositives;
            if (_averaging == AveragingMethod.Macro) { sum += fb; weights += 1; }
            else { sum += fb * count; weights += count; }
        }
        return weights > 0 ? NumOps.FromDouble(sum / weights) : NumOps.Zero;
    }

    private T ComputeBinary(Tensor<T> pred, Tensor<T> actual, T posLabel)
    {
        var counts = EvaluationTensorOps<T>.LabelCounts(pred, actual, posLabel);
        return NumOps.FromDouble(BinaryFromCounts(counts.TruePositives, counts.PredictedPositives, counts.ActualPositives));
    }

    private double BinaryFromCounts(int tp, int pp, int ap)
    {
        if (pp == 0 && ap == 0) return 1;
        double prec = pp > 0 ? (double)tp / pp : 0;
        double rec = ap > 0 ? (double)tp / ap : 0;
        // Precision and recall are both zero exactly when nothing was predicted correctly.
        if (tp == 0) return 0;
        double b2 = _beta * _beta;
        return (1 + b2) * prec * rec / (b2 * prec + rec);
    }

    private T ComputeMicro(Tensor<T> pred, Tensor<T> actual, Tensor<T> classes)
    {
        int totalTp = 0, totalPp = 0, totalAp = 0;
        foreach (T cls in classes)
        {
            var counts = EvaluationTensorOps<T>.LabelCounts(pred, actual, cls);
            totalPp += counts.PredictedPositives;
            totalAp += counts.ActualPositives;
            totalTp += counts.TruePositives;
        }
        double prec = totalPp > 0 ? (double)totalTp / totalPp : 0;
        double rec = totalAp > 0 ? (double)totalTp / totalAp : 0;
        if (prec + rec == 0) return NumOps.Zero;
        double b2 = _beta * _beta;
        return NumOps.FromDouble((1 + b2) * prec * rec / (b2 * prec + rec));
    }

    /// <inheritdoc />
    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual, int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var values = new List<T>(samples);
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();
        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = predArr[idx]; sa[i] = actArr[idx]; }
            values.Add(Compute(sp, sa));
        }
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return EvaluationTensorOps<T>.OrderStatistics(values, lo, hi);
    }
}
