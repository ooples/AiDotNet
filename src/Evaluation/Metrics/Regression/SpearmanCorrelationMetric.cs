using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Spearman's Rank Correlation Coefficient between predictions and actuals.
/// </summary>
/// <remarks>
/// <para>ρ = 1 - 6Σd²/(n(n²-1)) where d = rank difference</para>
/// <para><b>For Beginners:</b> Spearman correlation measures monotonic relationships:
/// <list type="bullet">
/// <item>Range: -1 to 1</item>
/// <item>1 = perfect positive monotonic relationship</item>
/// <item>-1 = perfect negative monotonic relationship</item>
/// <item>Non-parametric (uses ranks, not values)</item>
/// <item>Robust to outliers unlike Pearson correlation</item>
/// </list>
/// </para>
/// </remarks>
public class SpearmanCorrelationMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "SpearmanCorrelation";
    public string Category => "Regression";
    public string Description => "Spearman's Rank Correlation Coefficient.";
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

        // Compute ranks
        var predRanks = ComputeRanks(predictions);
        var actualRanks = ComputeRanks(actuals);

        // Compute Spearman correlation using Pearson correlation on ranks
        // This correctly handles ties (the d² formula assumes no ties)
        double meanPred = 0, meanActual = 0;
        for (int i = 0; i < n; i++)
        {
            meanPred += predRanks[i];
            meanActual += actualRanks[i];
        }
        meanPred /= n;
        meanActual /= n;

        double cov = 0, varPred = 0, varActual = 0;
        for (int i = 0; i < n; i++)
        {
            double dp = predRanks[i] - meanPred;
            double da = actualRanks[i] - meanActual;
            cov += dp * da;
            varPred += dp * dp;
            varActual += da * da;
        }

        // Guard against zero variance (constant ranks)
        if (varPred < 1e-12 || varActual < 1e-12)
            return NumOps.Zero;

        double rho = cov / Math.Sqrt(varPred * varActual);

        return NumOps.FromDouble(rho);
    }

    private double[] ComputeRanks(ReadOnlySpan<T> values)
    {
        int n = values.Length;
        var indexed = new (double value, int index)[n];
        for (int i = 0; i < n; i++)
        {
            indexed[i] = (NumOps.ToDouble(values[i]), i);
        }
        Array.Sort(indexed, (a, b) => a.value.CompareTo(b.value));

        var ranks = new double[n];
        int i2 = 0;
        while (i2 < n)
        {
            int start = i2;
            double value = indexed[start].value;

            // Find all tied values
            while (i2 < n && Math.Abs(indexed[i2].value - value) < 1e-10)
                i2++;

            // Assign average rank to all tied values
            double avgRank = (start + 1 + i2) / 2.0;
            for (int j = start; j < i2; j++)
            {
                ranks[indexed[j].index] = avgRank;
            }
        }

        return ranks;
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
