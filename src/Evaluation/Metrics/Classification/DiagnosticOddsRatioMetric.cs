using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Diagnostic Odds Ratio: (TP × TN) / (FP × FN).
/// </summary>
/// <remarks>
/// <para>DOR = (TP × TN) / (FP × FN) = (LR+) / (LR-)</para>
/// <para><b>For Beginners:</b> Diagnostic Odds Ratio:
/// <list type="bullet">
/// <item>Ratio of odds of positivity in disease to odds in non-disease</item>
/// <item>Range: 0 to infinity, higher is better</item>
/// <item>DOR = 1 means test has no discriminative power</item>
/// <item>DOR &gt; 1 means positive tests are more likely in disease</item>
/// </list>
/// </para>
/// <para><b>Medical interpretation:</b>
/// <list type="bullet">
/// <item>How many times more likely a positive result is in diseased vs healthy</item>
/// <item>Independent of disease prevalence</item>
/// <item>Used in meta-analyses of diagnostic tests</item>
/// </list>
/// </para>
/// </remarks>
public class DiagnosticOddsRatioMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "DiagnosticOddsRatio";
    public string Category => "Classification";
    public string Description => "Diagnostic Odds Ratio (TP×TN)/(FP×FN).";
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

        // Add 0.5 to each cell to handle zeros (Haldane-Anscombe correction)
        double tpAdj = tp + 0.5;
        double tnAdj = tn + 0.5;
        double fpAdj = fp + 0.5;
        double fnAdj = fn + 0.5;

        double dor = (tpAdj * tnAdj) / (fpAdj * fnAdj);
        return NumOps.FromDouble(dor);
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
        if (n == 0) return (NumOps.One, NumOps.One);
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
