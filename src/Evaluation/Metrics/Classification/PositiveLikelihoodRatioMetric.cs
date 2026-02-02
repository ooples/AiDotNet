using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Positive Likelihood Ratio (LR+): Sensitivity / (1 - Specificity) = TPR / FPR.
/// </summary>
/// <remarks>
/// <para>LR+ = Sensitivity / (1 - Specificity) = TPR / FPR</para>
/// <para><b>For Beginners:</b> Positive Likelihood Ratio:
/// <list type="bullet">
/// <item>How much more likely a positive test is in disease vs healthy</item>
/// <item>LR+ &gt; 10: Strong evidence for disease</item>
/// <item>LR+ 5-10: Moderate evidence</item>
/// <item>LR+ 2-5: Weak evidence</item>
/// <item>LR+ = 1: No diagnostic value</item>
/// </list>
/// </para>
/// <para><b>Clinical use:</b> Multiply pre-test odds by LR+ to get post-test odds.</para>
/// </remarks>
public class PositiveLikelihoodRatioMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "LR+";
    public string Category => "Classification";
    public string Description => "Positive Likelihood Ratio (Sensitivity / (1 - Specificity)).";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => default;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.One;

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

        double sensitivity = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
        double specificity = (tn + fp) > 0 ? (double)tn / (tn + fp) : 0;
        double fpr = 1 - specificity;

        if (fpr < 1e-10) return NumOps.FromDouble(double.MaxValue);
        return NumOps.FromDouble(sensitivity / fpr);
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
        if (n == 0) return (NumOps.One, NumOps.One);
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
