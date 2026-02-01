using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes False Positive Rate (FPR): proportion of actual negatives incorrectly predicted as positive.
/// </summary>
/// <remarks>
/// <para>FPR = FP / (FP + TN) = 1 - Specificity</para>
/// <para><b>For Beginners:</b> FPR answers: "Of all the actual negatives, how many did I incorrectly flag as positive?"
/// <list type="bullet">
/// <item>FPR = 0: Perfect - no false alarms</item>
/// <item>FPR = 0.1: 10% of actual negatives are incorrectly flagged as positive</item>
/// <item>FPR = 1: All negatives are incorrectly classified as positive</item>
/// </list>
/// Critical in spam detection (low FPR = fewer legitimate emails marked as spam)
/// and medical screening (low FPR = fewer healthy people incorrectly told they're sick).</para>
/// </remarks>
public class FalsePositiveRateMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "FPR";
    public string Category => "Classification";
    public string Description => "False Positive Rate (1 - Specificity).";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int fp = 0, tn = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            bool pred = NumOps.ToDouble(predictions[i]) >= 0.5;
            bool actual = NumOps.ToDouble(actuals[i]) >= 0.5;

            if (pred && !actual) fp++;
            else if (!pred && !actual) tn++;
        }

        int denominator = fp + tn;
        return denominator == 0 ? NumOps.Zero : NumOps.FromDouble((double)fp / denominator);
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
