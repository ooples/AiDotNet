using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Brier Score: mean squared error of probability predictions.
/// </summary>
/// <remarks>
/// <para>
/// Brier Score = (1/N) * Σ(p_i - y_i)²
/// </para>
/// <para>
/// <b>For Beginners:</b> Brier score measures how close predicted probabilities are to actual outcomes.
/// <list type="bullet">
/// <item>Brier = 0: Perfect predictions (probabilities match outcomes exactly)</item>
/// <item>Brier = 0.25: Random guessing for binary classification</item>
/// <item>Brier = 1: Completely wrong (predicting 0% for all actual positives)</item>
/// </list>
/// </para>
/// <para>
/// <b>Advantages:</b> Sensitive to calibration, proper scoring rule, penalizes overconfidence.
/// </para>
/// </remarks>
public class BrierScoreMetric<T> : IProbabilisticClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "BrierScore";
    public string Category => "Classification";
    public string Description => "Mean squared error of probability predictions.";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => true;
    public bool SupportsMultiClass => true;

    public T Compute(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses = 2)
    {
        if (probabilities.Length == 0) return NumOps.Zero;

        if (numClasses == 2)
        {
            if (probabilities.Length != actuals.Length)
                throw new ArgumentException("For binary, probabilities and actuals must have same length.");
            return ComputeBinaryBrier(probabilities, actuals);
        }
        else
        {
            return ComputeMultiClassBrier(probabilities, actuals, numClasses);
        }
    }

    private T ComputeBinaryBrier(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals)
    {
        int n = probs.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            double p = NumOps.ToDouble(probs[i]);
            double y = NumOps.ToDouble(actuals[i]);
            double diff = p - y;
            sum += diff * diff;
        }
        return NumOps.FromDouble(sum / n);
    }

    private T ComputeMultiClassBrier(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals, int numClasses)
    {
        int n = actuals.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            int trueClass = (int)Math.Round(NumOps.ToDouble(actuals[i]));
            for (int c = 0; c < numClasses; c++)
            {
                double p = NumOps.ToDouble(probs[i * numClasses + c]);
                double y = (c == trueClass) ? 1.0 : 0.0;
                double diff = p - y;
                sum += diff * diff;
            }
        }
        return NumOps.FromDouble(sum / (n * numClasses));
    }

    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals,
        int numClasses = 2, ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.BCaBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        var value = Compute(probabilities, actuals, numClasses);
        var (lower, upper) = BootstrapCI(probabilities, actuals, numClasses, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> probs, ReadOnlySpan<T> actuals, int numClasses,
        int samples, double conf, int? seed)
    {
        int n = actuals.Length;
        if (n == 0) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var values = new double[samples];
        var probArr = probs.ToArray();
        var actArr = actuals.ToArray();
        int probsPerSample = numClasses == 2 ? 1 : numClasses;

        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n * probsPerSample];
            var sa = new T[n];
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sa[i] = actArr[idx];
                for (int c = 0; c < probsPerSample; c++)
                    sp[i * probsPerSample + c] = probArr[idx * probsPerSample + c];
            }
            values[b] = NumOps.ToDouble(Compute(sp, sa, numClasses));
        }
        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
