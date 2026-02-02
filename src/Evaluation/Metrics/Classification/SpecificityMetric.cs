using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes specificity (true negative rate): the proportion of actual negatives correctly identified.
/// </summary>
/// <remarks>
/// <para>
/// Specificity = TN / (TN + FP) = True Negatives / Actual Negatives
/// </para>
/// <para>
/// <b>For Beginners:</b> Specificity answers: "Of all actual negatives, how many did the model correctly identify?"
/// High specificity means few false positives. A spam filter with 99% specificity means it correctly
/// lets through 99% of legitimate emails.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpecificityMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly T _positiveLabel;

    public string Name => "Specificity";
    public string Category => "Classification";
    public string Description => "Proportion of actual negatives correctly identified (TN / (TN + FP)).";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public SpecificityMetric(T? positiveLabel = default)
    {
        _positiveLabel = positiveLabel ?? NumOps.One;
    }

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");

        if (predictions.Length == 0) return NumOps.Zero;

        double positiveLabelValue = NumOps.ToDouble(_positiveLabel);
        int trueNegatives = 0;
        int actualNegatives = 0;

        for (int i = 0; i < actuals.Length; i++)
        {
            bool isActualPositive = Math.Abs(NumOps.ToDouble(actuals[i]) - positiveLabelValue) < 1e-10;
            bool isPredictedPositive = Math.Abs(NumOps.ToDouble(predictions[i]) - positiveLabelValue) < 1e-10;

            if (!isActualPositive)
            {
                actualNegatives++;
                if (!isPredictedPositive) trueNegatives++;
            }
        }

        return actualNegatives > 0 ? NumOps.FromDouble((double)trueNegatives / actualNegatives) : NumOps.One;
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
        if (n == 0) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var values = new double[samples];
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();

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
