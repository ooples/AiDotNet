using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes R² (coefficient of determination): proportion of variance explained by the model.
/// </summary>
/// <remarks>
/// <para>R² = 1 - (SS_res / SS_tot) = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²</para>
/// <para><b>For Beginners:</b> R² tells you what percentage of the variance in your target is
/// explained by the model.
/// <list type="bullet">
/// <item>R² = 1: Perfect fit (model explains 100% of variance)</item>
/// <item>R² = 0: Model is no better than predicting the mean</item>
/// <item>R² &lt; 0: Model is worse than predicting the mean</item>
/// </list>
/// </para>
/// </remarks>
public class R2ScoreMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "R2Score";
    public string Category => "Regression";
    public string Description => "Coefficient of determination - proportion of variance explained.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => default; // Can be negative
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        // Calculate mean of actuals
        double sumActuals = 0;
        for (int i = 0; i < actuals.Length; i++)
            sumActuals += NumOps.ToDouble(actuals[i]);
        double mean = sumActuals / actuals.Length;

        // Calculate SS_res and SS_tot
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < actuals.Length; i++)
        {
            double actual = NumOps.ToDouble(actuals[i]);
            double pred = NumOps.ToDouble(predictions[i]);
            double diffRes = actual - pred;
            double diffTot = actual - mean;
            ssRes += diffRes * diffRes;
            ssTot += diffTot * diffTot;
        }

        if (Math.Abs(ssTot) < 1e-10) return NumOps.One; // All actuals are the same
        return NumOps.FromDouble(1 - ssRes / ssTot);
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
        if (n == 0) return (NumOps.Zero, NumOps.One);
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
