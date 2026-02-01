using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Mean Bias Error (MBE): average signed error showing systematic over/under-prediction.
/// </summary>
/// <remarks>
/// <para>MBE = (1/N) * Σ(ŷ_i - y_i)</para>
/// <para><b>For Beginners:</b> Mean Bias Error tells you if your model systematically over or under-predicts:
/// <list type="bullet">
/// <item>MBE = 0: No systematic bias (over and under-predictions cancel out)</item>
/// <item>MBE &gt; 0: Model tends to over-predict (predictions are too high on average)</item>
/// <item>MBE &lt; 0: Model tends to under-predict (predictions are too low on average)</item>
/// </list>
/// Note: MBE alone doesn't tell you about accuracy - use alongside MAE or RMSE.
/// A model could have MBE ≈ 0 but terrible accuracy if errors are random but balanced.</para>
/// </remarks>
public class MeanBiasErrorMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "MBE";
    public string Category => "Regression";
    public string Description => "Mean Bias Error showing systematic over/under-prediction.";
    public MetricDirection Direction => MetricDirection.TargetValue;
    public T? MinValue => default; // Can be negative
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double sumBias = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            sumBias += NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(actuals[i]);
        }

        return NumOps.FromDouble(sumBias / predictions.Length);
    }

    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.BCaBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual, int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.Zero);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
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
