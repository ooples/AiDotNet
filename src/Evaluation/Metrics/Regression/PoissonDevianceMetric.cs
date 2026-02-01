using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Poisson Deviance for count data regression.
/// </summary>
/// <remarks>
/// <para>Poisson Deviance = 2 × Σ(y × log(y/ŷ) - (y - ŷ)) for y &gt; 0</para>
/// <para><b>For Beginners:</b> Poisson Deviance is designed for count data:
/// <list type="bullet">
/// <item>Perfect for predicting counts (visitors, purchases, events)</item>
/// <item>Handles the discrete nature of count data</item>
/// <item>Penalizes relative errors rather than absolute errors</item>
/// <item>Special case of Tweedie deviance with power = 1</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Predicting number of events (clicks, purchases, calls)</item>
/// <item>When predictions should always be positive</item>
/// <item>When variance increases with the mean</item>
/// </list>
/// </para>
/// </remarks>
public class PoissonDevianceMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "PoissonDeviance";
    public string Category => "Regression";
    public string Description => "Poisson Deviance for count data regression.";
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

        double totalDeviance = 0;
        int validCount = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            double y = Math.Max(0, NumOps.ToDouble(actuals[i]));
            double mu = Math.Max(1e-10, NumOps.ToDouble(predictions[i]));

            double deviance;
            if (y > 0)
            {
                deviance = 2 * (y * Math.Log(y / mu) - (y - mu));
            }
            else
            {
                deviance = 2 * mu;
            }

            if (!double.IsNaN(deviance) && !double.IsInfinity(deviance))
            {
                totalDeviance += deviance;
                validCount++;
            }
        }

        return validCount > 0 ? NumOps.FromDouble(totalDeviance / validCount) : NumOps.Zero;
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
