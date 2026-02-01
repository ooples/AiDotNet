using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Gini Coefficient (normalized): 2 * AUC - 1, measures discriminative ability.
/// </summary>
/// <remarks>
/// <para>Gini = 2 * AUC - 1</para>
/// <para><b>For Beginners:</b> Gini coefficient is directly related to AUC:
/// <list type="bullet">
/// <item>Gini = 0: Random classifier (AUC = 0.5)</item>
/// <item>Gini = 1: Perfect classifier (AUC = 1.0)</item>
/// <item>Gini = -1: Perfectly wrong classifier (AUC = 0.0)</item>
/// </list>
/// Common in credit scoring and insurance modeling. Sometimes called the "Gini index" (not to be confused
/// with Gini impurity used in decision trees).</para>
/// </remarks>
public class GiniCoefficientMetric<T> : IProbabilisticClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly AUCROCMetric<T> _aucMetric = new();

    public string Name => "GiniCoefficient";
    public string Category => "Classification";
    public string Description => "Gini coefficient (2*AUC - 1) measuring discriminative power.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.FromDouble(-1.0);
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => true;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses = 2)
    {
        var auc = _aucMetric.Compute(probabilities, actuals, numClasses);
        double aucVal = NumOps.ToDouble(auc);
        return NumOps.FromDouble(2 * aucVal - 1);
    }

    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.BCaBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        var value = Compute(probabilities, actuals, numClasses);
        var (lower, upper) = BootstrapCI(probabilities, actuals, numClasses, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> prob, ReadOnlySpan<T> actual, int numClasses, int samples, double conf, int? seed)
    {
        int n = prob.Length;
        if (n == 0) return (NumOps.FromDouble(-1), NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var values = new double[samples];
        var probArr = prob.ToArray(); var actArr = actual.ToArray();
        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = probArr[idx]; sa[i] = actArr[idx]; }
            values[b] = NumOps.ToDouble(Compute(sp, sa, numClasses));
        }
        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
