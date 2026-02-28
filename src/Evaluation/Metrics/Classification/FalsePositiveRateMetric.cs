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
    private readonly T _positiveLabel;

    public string Name => "FPR";
    public string Category => "Classification";
    public string Description => "False Positive Rate (1 - Specificity).";
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public FalsePositiveRateMetric()
    {
        _positiveLabel = NumOps.One;
    }

    public FalsePositiveRateMetric(T positiveLabel)
    {
        _positiveLabel = positiveLabel;
    }

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        double positiveLabelValue = NumOps.ToDouble(_positiveLabel);
        int fp = 0, tn = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            bool isActualPositive = Math.Abs(NumOps.ToDouble(actuals[i]) - positiveLabelValue) < 1e-10;
            bool isPredictedPositive = Math.Abs(NumOps.ToDouble(predictions[i]) - positiveLabelValue) < 1e-10;

            if (!isActualPositive)
            {
                if (isPredictedPositive) fp++;
                else tn++;
            }
        }

        int denominator = fp + tn;
        return denominator == 0 ? NumOps.Zero : NumOps.FromDouble((double)fp / denominator);
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
        if (n == 0) return (NumOps.Zero, NumOps.Zero);
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
