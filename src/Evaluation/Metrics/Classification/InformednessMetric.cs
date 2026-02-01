using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Informedness (Youden's J statistic): TPR + TNR - 1 = Recall + Specificity - 1.
/// </summary>
/// <remarks>
/// <para>Informedness = TPR + TNR - 1 = Sensitivity + Specificity - 1</para>
/// <para><b>For Beginners:</b> Informedness measures how "informed" the classifier is:
/// <list type="bullet">
/// <item>+1: Perfect classifier</item>
/// <item>0: Random classifier (no better than chance)</item>
/// <item>-1: Perfectly wrong classifier</item>
/// </list>
/// Also known as Youden's J statistic. It's the optimal threshold point on a ROC curve and
/// considers both classes equally important.</para>
/// </remarks>
public class InformednessMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "Informedness";
    public string Category => "Classification";
    public string Description => "Youden's J statistic (TPR + TNR - 1).";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.FromDouble(-1.0);
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            bool pred = NumOps.ToDouble(predictions[i]) >= 0.5;
            bool actual = NumOps.ToDouble(actuals[i]) >= 0.5;

            if (pred && actual) tp++;
            else if (!pred && !actual) tn++;
            else if (pred && !actual) fp++;
            else fn++;
        }

        double tpr = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
        double tnr = (tn + fp) > 0 ? (double)tn / (tn + fp) : 0;

        return NumOps.FromDouble(tpr + tnr - 1);
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
        if (n == 0) return (NumOps.FromDouble(-1), NumOps.One);
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
