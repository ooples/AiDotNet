using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Median Absolute Error (MedAE): the median of all absolute errors.
/// </summary>
/// <remarks>
/// <para>MedAE = median(|y_i - ŷ_i|)</para>
/// <para><b>For Beginners:</b> Median Absolute Error is even more robust to outliers than MAE:
/// <list type="bullet">
/// <item>Uses the median instead of mean, so outliers have minimal impact</item>
/// <item>Represents the "typical" error - half your predictions are better, half are worse</item>
/// <item>Particularly useful when your data has heavy-tailed error distributions</item>
/// </list>
/// Compare MedAE vs MAE: if MedAE is much smaller than MAE, you have outlier issues.</para>
/// </remarks>
public class MedianAbsoluteErrorMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "MedAE";
    public string Category => "Regression";
    public string Description => "Median Absolute Error - robust to outliers.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var errors = new double[predictions.Length];
        for (int i = 0; i < predictions.Length; i++)
        {
            errors[i] = Math.Abs(NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(predictions[i]));
        }

        Array.Sort(errors);
        int n = errors.Length;
        double median = n % 2 == 1
            ? errors[n / 2]
            : (errors[n / 2 - 1] + errors[n / 2]) / 2.0;

        return NumOps.FromDouble(median);
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
        if (n == 0) return (NumOps.Zero, NumOps.Zero);
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
