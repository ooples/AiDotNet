using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Log-Cosh Loss: mean of log(cosh(y - ŷ)).
/// </summary>
/// <remarks>
/// <para>Log-Cosh = (1/N) × Σ log(cosh(y - ŷ))</para>
/// <para><b>For Beginners:</b> Log-Cosh loss combines benefits of MSE and MAE:
/// <list type="bullet">
/// <item>Behaves like MAE for large errors (robust to outliers)</item>
/// <item>Behaves like MSE for small errors (smooth gradient)</item>
/// <item>Always positive and convex</item>
/// <item>Has continuous second derivative (unlike Huber)</item>
/// </list>
/// </para>
/// <para><b>Advantages over alternatives:</b>
/// <list type="bullet">
/// <item>vs MSE: More robust to outliers</item>
/// <item>vs MAE: Smoother gradient near zero</item>
/// <item>vs Huber: No hyperparameter to tune</item>
/// </list>
/// </para>
/// </remarks>
public class LogCoshLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "LogCoshLoss";
    public string Category => "Regression";
    public string Description => "Log-Cosh Loss (smooth approximation of MAE).";
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

            // log(cosh(x)) ≈ x²/2 for small x, |x| - log(2) for large x
            // Use numerically stable computation
            double absError = Math.Abs(error);
            double logCosh;
            if (absError < 20)
            {
                logCosh = Math.Log(Math.Cosh(error));
            }
            else
            {
                // For large values: log(cosh(x)) ≈ |x| - log(2)
                logCosh = absError - Math.Log(2);
            }

            totalLoss += logCosh;
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
