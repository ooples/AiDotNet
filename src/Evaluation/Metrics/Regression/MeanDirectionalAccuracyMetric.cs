using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Mean Directional Accuracy (MDA): fraction of correctly predicted directions.
/// </summary>
/// <remarks>
/// <para>MDA = (1/N) × Σ I(sign(y_t - y_{t-1}) = sign(ŷ_t - ŷ_{t-1}))</para>
/// <para><b>For Beginners:</b> MDA measures if you predicted the right direction:
/// <list type="bullet">
/// <item>Did the model predict "up" when actual went up?</item>
/// <item>Range: 0 to 1, higher is better</item>
/// <item>0.5 = random guessing</item>
/// <item>Important in trading/forecasting where direction matters more than magnitude</item>
/// </list>
/// </para>
/// </remarks>
public class MeanDirectionalAccuracyMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "MDA";
    public string Category => "Regression";
    public string Description => "Mean Directional Accuracy (fraction of correct direction predictions).";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length < 2) return NumOps.FromDouble(0.5);

        int correct = 0;
        int total = 0;
        for (int i = 1; i < predictions.Length; i++)
        {
            double actualChange = NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(actuals[i - 1]);
            double predChange = NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(predictions[i - 1]);

            // Check if directions match (both positive, both negative, or both zero)
            if ((actualChange > 0 && predChange > 0) ||
                (actualChange < 0 && predChange < 0) ||
                (Math.Abs(actualChange) < 1e-10 && Math.Abs(predChange) < 1e-10))
            {
                correct++;
            }
            total++;
        }

        return total > 0 ? NumOps.FromDouble((double)correct / total) : NumOps.FromDouble(0.5);
    }

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
        if (n < 2) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var values = new double[samples];
        var predArr = pred.ToArray(); var actArr = actual.ToArray();
        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = predArr[idx]; sa[i] = actArr[idx]; }
            values[b] = NumOps.ToDouble(Compute(sp, sa));
        }
        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
