using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Area Under the Precision-Recall Curve (AUC-PR).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AUC-PR measures the trade-off between precision and recall:
/// <list type="bullet">
/// <item>Better than AUC-ROC for imbalanced datasets</item>
/// <item>Focuses on the positive class performance</item>
/// <item>Range: 0 to 1, higher is better</item>
/// <item>Baseline depends on class imbalance (unlike ROC's 0.5)</item>
/// </list>
/// </para>
/// <para><b>When to use AUC-PR vs AUC-ROC:</b>
/// <list type="bullet">
/// <item>AUC-PR: When positive class is rare and important (fraud, disease)</item>
/// <item>AUC-ROC: When both classes are equally important</item>
/// </list>
/// </para>
/// </remarks>
public class AUCPRMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "AUC-PR";
    /// <inheritdoc />
    public string Category => "Classification";
    /// <inheritdoc />
    public string Description => "Area Under the Precision-Recall Curve.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => true;
    /// <inheritdoc />
    public bool SupportsMultiClass => false;

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        // Average precision: precision at each distinct threshold weighted by the recall gained there, so tied
        // scores share one threshold (scikit-learn's average_precision_score).
        var scores = EvaluationTensorOps<T>.ToTensor(predictions);
        var positiveMask = EvaluationTensorOps<T>.EqualMask(EvaluationTensorOps<T>.ToTensor(actuals), NumOps.One);
        return EvaluationTensorOps<T>.AveragePrecision(scores, positiveMask);
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
        var predArr = pred.ToArray(); var actArr = actual.ToArray();
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
