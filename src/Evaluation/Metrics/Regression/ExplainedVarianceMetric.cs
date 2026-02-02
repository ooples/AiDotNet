using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Explained Variance Score: proportion of variance in target explained by predictions.
/// </summary>
/// <remarks>
/// <para>EV = 1 - Var(y - ŷ) / Var(y)</para>
/// <para><b>For Beginners:</b> Similar to R² but doesn't penalize bias. EV = 1 means perfect variance
/// explanation, EV = 0 means no better than mean prediction. Can differ from R² if predictions are biased.</para>
/// </remarks>
public class ExplainedVarianceMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "ExplainedVariance";
    public string Category => "Regression";
    public string Description => "Proportion of variance explained, ignoring bias.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => default;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int n = predictions.Length;

        // Calculate means
        double sumActuals = 0, sumResiduals = 0;
        for (int i = 0; i < n; i++)
        {
            sumActuals += NumOps.ToDouble(actuals[i]);
            sumResiduals += NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(predictions[i]);
        }
        double meanActuals = sumActuals / n;
        double meanResiduals = sumResiduals / n;

        // Calculate variances
        double varActuals = 0, varResiduals = 0;
        for (int i = 0; i < n; i++)
        {
            double actual = NumOps.ToDouble(actuals[i]);
            double residual = actual - NumOps.ToDouble(predictions[i]);
            double diffActuals = actual - meanActuals;
            double diffResiduals = residual - meanResiduals;
            varActuals += diffActuals * diffActuals;
            varResiduals += diffResiduals * diffResiduals;
        }

        if (Math.Abs(varActuals) < 1e-10) return NumOps.One;
        return NumOps.FromDouble(1 - varResiduals / varActuals);
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
        if (n == 0) return (NumOps.Zero, NumOps.One);
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
