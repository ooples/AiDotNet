using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes True Negative Rate (TNR): same as Specificity, TN / (TN + FP).
/// </summary>
/// <remarks>
/// <para>TNR = TN / (TN + FP) = Specificity = 1 - FPR</para>
/// <para><b>For Beginners:</b> TNR answers: "Of all actual negatives, what proportion did we correctly identify?"
/// This is the same as Specificity but named from a different perspective.
/// <list type="bullet">
/// <item>TNR = 1: All negatives correctly identified</item>
/// <item>TNR = 0.9: 90% of negatives correctly identified, 10% false alarms</item>
/// </list>
/// </para>
/// </remarks>
public class TrueNegativeRateMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "TNR";
    public string Category => "Classification";
    public string Description => "True Negative Rate (Specificity).";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int tn = 0, fp = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            bool pred = NumOps.ToDouble(predictions[i]) >= 0.5;
            bool actual = NumOps.ToDouble(actuals[i]) >= 0.5;

            if (!pred && !actual) tn++;
            else if (pred && !actual) fp++;
        }

        int denominator = tn + fp;
        return denominator == 0 ? NumOps.One : NumOps.FromDouble((double)tn / denominator);
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
