using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Quantile Loss (Pinball Loss) for quantile regression.
/// </summary>
/// <remarks>
/// <para>Quantile Loss = (1/N) × Σ max(τ(y - ŷ), (τ - 1)(y - ŷ))</para>
/// <para><b>For Beginners:</b> Quantile loss is used when you want to predict a specific percentile:
/// <list type="bullet">
/// <item>τ = 0.5: Median (equivalent to MAE)</item>
/// <item>τ = 0.9: 90th percentile (over-predictions penalized less)</item>
/// <item>τ = 0.1: 10th percentile (under-predictions penalized less)</item>
/// </list>
/// </para>
/// <para><b>Use cases:</b>
/// <list type="bullet">
/// <item>Risk assessment (predict 95th percentile of losses)</item>
/// <item>Inventory planning (predict 99th percentile of demand)</item>
/// <item>Uncertainty quantification (predict multiple quantiles)</item>
/// </list>
/// </para>
/// </remarks>
public class QuantileLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _quantile;

    public QuantileLossMetric(double quantile = 0.5)
    {
        if (quantile <= 0 || quantile >= 1)
            throw new ArgumentException("Quantile must be between 0 and 1 (exclusive).");
        _quantile = quantile;
    }

    public string Name => $"QuantileLoss(τ={_quantile})";
    public string Category => "Regression";
    public string Description => $"Quantile Loss (Pinball Loss) at τ={_quantile}.";
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

        double totalLoss = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double y = NumOps.ToDouble(actuals[i]);
            double yHat = NumOps.ToDouble(predictions[i]);
            double error = y - yHat;

            // Pinball loss: asymmetric penalty
            double loss = error >= 0 ? _quantile * error : (_quantile - 1) * error;
            totalLoss += loss;
        }

        return NumOps.FromDouble(totalLoss / predictions.Length);
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
