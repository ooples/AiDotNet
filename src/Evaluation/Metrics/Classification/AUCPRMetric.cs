using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Area Under the Precision-Recall Curve (AUC-PR).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AUC-PR measures the trade-off between precision and recall:
/// <list type="bullet">
/// <item>Better than AUC-ROC for imbalanced datasets</item>
/// <item>Focuses on the positive class performance</item>
/// <item>Range: 0 to 1, higher is better</item>
/// <item>Baseline depends on class imbalance (unlike ROC's 0.5)</item>
/// </list>
/// </para>
/// <para><b>When to use AUC-PR vs AUC-ROC:</b>
/// <list type="bullet">
/// <item>AUC-PR: When positive class is rare and important (fraud, disease)</item>
/// <item>AUC-ROC: When both classes are equally important</item>
/// </list>
/// </para>
/// </remarks>
public class AUCPRMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "AUC-PR";
    public string Category => "Classification";
    public string Description => "Area Under the Precision-Recall Curve.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => true;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int n = predictions.Length;
        var sorted = new (double prob, int label)[n];
        for (int i = 0; i < n; i++)
        {
            sorted[i] = (NumOps.ToDouble(predictions[i]), NumOps.ToDouble(actuals[i]) >= 0.5 ? 1 : 0);
        }
        Array.Sort(sorted, (a, b) => b.prob.CompareTo(a.prob));

        int totalPositives = sorted.Count(s => s.label == 1);
        if (totalPositives == 0) return NumOps.Zero;

        double auc = 0;
        double prevRecall = 0;
        int tp = 0;
        int fp = 0;

        for (int i = 0; i < n; i++)
        {
            if (sorted[i].label == 1) tp++;
            else fp++;

            double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
            double recall = (double)tp / totalPositives;

            if (recall > prevRecall)
            {
                auc += precision * (recall - prevRecall);
                prevRecall = recall;
            }
        }

        return NumOps.FromDouble(auc);
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
