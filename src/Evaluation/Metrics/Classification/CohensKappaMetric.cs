using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Cohen's Kappa: agreement measure that accounts for chance agreement.
/// </summary>
/// <remarks>
/// <para>
/// Kappa = (p_o - p_e) / (1 - p_e)
/// where p_o = observed agreement, p_e = expected agreement by chance
/// </para>
/// <para>
/// <b>For Beginners:</b> Kappa measures how much better the model is than random guessing.
/// <list type="bullet">
/// <item>Kappa = 1: Perfect agreement</item>
/// <item>Kappa = 0: No better than chance</item>
/// <item>Kappa &lt; 0: Worse than chance</item>
/// </list>
/// </para>
/// <para>
/// <b>Interpretation guidelines (Landis &amp; Koch):</b>
/// &lt;0: Poor, 0-0.2: Slight, 0.2-0.4: Fair, 0.4-0.6: Moderate, 0.6-0.8: Substantial, 0.8-1: Almost perfect
/// </para>
/// </remarks>
public class CohensKappaMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "CohensKappa";
    /// <inheritdoc />
    public string Category => "Classification";
    /// <inheritdoc />
    public string Description => "Agreement measure accounting for chance, ranging from -1 to 1.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.FromDouble(-1.0);
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => true;

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var classes = EvaluationTensorOps<T>.UniqueLabels(actual, pred);
        int n = predictions.Length;

        // Observed agreement: the trace of the confusion matrix
        double po = (double)EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.PairEqualMask(pred, actual)) / n;

        // Expected agreement: sum over classes of (row total / n) * (column total / n)
        double pe = 0;
        foreach (T cls in classes)
        {
            var counts = EvaluationTensorOps<T>.LabelCounts(pred, actual, cls);
            pe += ((double)counts.ActualPositives / n) * ((double)counts.PredictedPositives / n);
        }

        if (Math.Abs(1 - pe) < 1e-10) return NumOps.One;
        return NumOps.FromDouble((po - pe) / (1 - pe));
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
        if (n == 0) return (NumOps.FromDouble(-1), NumOps.One);
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
