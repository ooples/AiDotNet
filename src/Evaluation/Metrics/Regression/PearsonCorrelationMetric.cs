using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Pearson Correlation Coefficient between predictions and actuals.
/// </summary>
/// <remarks>
/// <para>r = Cov(y, ŷ) / (σ_y × σ_ŷ)</para>
/// <para><b>For Beginners:</b> Pearson correlation measures linear relationships:
/// <list type="bullet">
/// <item>Range: -1 to 1</item>
/// <item>1 = perfect positive linear relationship</item>
/// <item>-1 = perfect negative linear relationship</item>
/// <item>0 = no linear relationship</item>
/// <item>Note: R² = r² for simple regression</item>
/// </list>
/// </para>
/// </remarks>
public class PearsonCorrelationMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "PearsonCorrelation";
    public string Category => "Regression";
    public string Description => "Pearson Correlation Coefficient.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.FromDouble(-1.0);
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length < 2) return NumOps.Zero;

        int n = predictions.Length;

        // Compute means
        double sumPred = 0, sumActual = 0;
        for (int i = 0; i < n; i++)
        {
            sumPred += NumOps.ToDouble(predictions[i]);
            sumActual += NumOps.ToDouble(actuals[i]);
        }
        double meanPred = sumPred / n;
        double meanActual = sumActual / n;

        // Compute covariance and standard deviations
        double covariance = 0;
        double varPred = 0;
        double varActual = 0;
        for (int i = 0; i < n; i++)
        {
            double diffPred = NumOps.ToDouble(predictions[i]) - meanPred;
            double diffActual = NumOps.ToDouble(actuals[i]) - meanActual;
            covariance += diffPred * diffActual;
            varPred += diffPred * diffPred;
            varActual += diffActual * diffActual;
        }

        double stdPred = Math.Sqrt(varPred);
        double stdActual = Math.Sqrt(varActual);

        if (stdPred < 1e-10 || stdActual < 1e-10)
            return NumOps.Zero;

        double r = covariance / (stdPred * stdActual);
        return NumOps.FromDouble(r);
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
        if (n < 2) return (NumOps.FromDouble(-1), NumOps.One);
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
