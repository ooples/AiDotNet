using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Huber Loss: a robust loss function that is less sensitive to outliers than MSE.
/// </summary>
/// <remarks>
/// <para>
/// Huber Loss = (1/N) * Σ L_δ(y, ŷ) where:
/// <list type="bullet">
/// <item>L_δ = 0.5 * (y - ŷ)² if |y - ŷ| ≤ δ (quadratic for small errors)</item>
/// <item>L_δ = δ * (|y - ŷ| - 0.5 * δ) if |y - ŷ| > δ (linear for large errors)</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Huber loss combines the best of MSE and MAE:
/// <list type="bullet">
/// <item>For small errors (≤ δ): Uses squared error like MSE (smooth gradient)</item>
/// <item>For large errors (> δ): Uses linear error like MAE (robust to outliers)</item>
/// </list>
/// The delta parameter controls where the transition happens. Common values are 1.0 or 1.35.</para>
/// </remarks>
public class HuberLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _delta;

    public string Name => "HuberLoss";
    public string Category => "Regression";
    public string Description => "Robust loss function combining MSE and MAE.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Initializes the Huber Loss metric.
    /// </summary>
    /// <param name="delta">Threshold at which to switch from quadratic to linear. Default is 1.0.</param>
    public HuberLossMetric(double delta = 1.0)
    {
        _delta = Math.Max(0.001, delta);
    }

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double totalLoss = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double actual = NumOps.ToDouble(actuals[i]);
            double pred = NumOps.ToDouble(predictions[i]);
            double error = Math.Abs(actual - pred);

            if (error <= _delta)
            {
                // Quadratic region
                totalLoss += 0.5 * error * error;
            }
            else
            {
                // Linear region
                totalLoss += _delta * (error - 0.5 * _delta);
            }
        }

        return NumOps.FromDouble(totalLoss / predictions.Length);
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
