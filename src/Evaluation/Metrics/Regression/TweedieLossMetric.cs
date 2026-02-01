using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Tweedie Deviance Loss for regression with power parameter.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Tweedie loss is flexible for different data types:
/// <list type="bullet">
/// <item>Power = 0: Normal distribution (like MSE)</item>
/// <item>Power = 1: Poisson distribution (count data)</item>
/// <item>Power = 2: Gamma distribution (positive continuous)</item>
/// <item>Power = 3: Inverse Gaussian distribution</item>
/// <item>1 &lt; Power &lt; 2: Compound Poisson-Gamma (insurance claims)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Insurance claim prediction (zeros and positives)</item>
/// <item>Sales forecasting with many zeros</item>
/// <item>Any data with mixed zero and positive values</item>
/// </list>
/// </para>
/// </remarks>
public class TweedieLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _power;

    public TweedieLossMetric(double power = 1.5)
    {
        if (power < 0 || (power > 0 && power < 1))
            throw new ArgumentException("Power must be 0 or >= 1.");
        _power = power;
    }

    public string Name => $"TweedieLoss(p={_power})";
    public string Category => "Regression";
    public string Description => $"Tweedie Deviance Loss with power parameter {_power}.";
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
        int validCount = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            double y = NumOps.ToDouble(actuals[i]);
            double mu = Math.Max(1e-10, NumOps.ToDouble(predictions[i]));

            double loss;
            if (Math.Abs(_power) < 1e-10) // Normal (p = 0)
            {
                loss = (y - mu) * (y - mu);
            }
            else if (Math.Abs(_power - 1) < 1e-10) // Poisson (p = 1)
            {
                loss = y > 0 ? 2 * (y * Math.Log(y / mu) - (y - mu)) : 2 * mu;
            }
            else if (Math.Abs(_power - 2) < 1e-10) // Gamma (p = 2)
            {
                loss = y > 0 ? 2 * (-Math.Log(y / mu) + (y - mu) / mu) : 0;
            }
            else // General case
            {
                double a = Math.Max(1e-10, Math.Pow(y, 2 - _power)) / ((1 - _power) * (2 - _power));
                double b = y * Math.Pow(mu, 1 - _power) / (1 - _power);
                double c = Math.Pow(mu, 2 - _power) / (2 - _power);
                loss = 2 * (a - b + c);
            }

            if (!double.IsNaN(loss) && !double.IsInfinity(loss))
            {
                totalLoss += loss;
                validCount++;
            }
        }

        return validCount > 0 ? NumOps.FromDouble(totalLoss / validCount) : NumOps.Zero;
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
