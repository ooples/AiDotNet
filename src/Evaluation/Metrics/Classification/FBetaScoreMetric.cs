using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes F-beta score: weighted harmonic mean of precision and recall.
/// </summary>
/// <remarks>
/// <para>
/// F_beta = (1 + beta²) * (precision * recall) / (beta² * precision + recall)
/// </para>
/// <para>
/// <b>For Beginners:</b> F-beta lets you weight precision vs recall:
/// <list type="bullet">
/// <item>beta = 1: F1 score (equal weight)</item>
/// <item>beta = 2: F2 score (recall twice as important as precision)</item>
/// <item>beta = 0.5: F0.5 score (precision twice as important as recall)</item>
/// </list>
/// </para>
/// </remarks>
public class FBetaScoreMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly T _positiveLabel;
    private readonly double _beta;
    private readonly AveragingMethod _averaging;

    public string Name => $"F{_beta:F1}Score";
    public string Category => "Classification";
    public string Description => $"F-beta score with beta={_beta}, weighting recall {(_beta > 1 ? "more" : _beta < 1 ? "less" : "equally")} than precision.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => NumOps.Zero;
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => true;

    public FBetaScoreMetric(double beta = 1.0, T? positiveLabel = default, AveragingMethod averaging = AveragingMethod.Binary)
    {
        _beta = beta;
        _positiveLabel = positiveLabel ?? NumOps.One;
        _averaging = averaging;
    }

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        if (_averaging == AveragingMethod.Binary || _averaging == AveragingMethod.None)
            return ComputeBinary(predictions, actuals, _positiveLabel);

        var classes = GetClasses(actuals);
        if (_averaging == AveragingMethod.Micro)
            return ComputeMicro(predictions, actuals, classes);

        double sum = 0, weights = 0;
        foreach (var cls in classes)
        {
            var fb = NumOps.ToDouble(ComputeBinary(predictions, actuals, NumOps.FromDouble(cls)));
            int count = CountClass(actuals, cls);
            if (_averaging == AveragingMethod.Macro) { sum += fb; weights += 1; }
            else { sum += fb * count; weights += count; }
        }
        return weights > 0 ? NumOps.FromDouble(sum / weights) : NumOps.Zero;
    }

    private T ComputeBinary(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual, T posLabel)
    {
        double posVal = NumOps.ToDouble(posLabel);
        int tp = 0, pp = 0, ap = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            bool isPred = Math.Abs(NumOps.ToDouble(pred[i]) - posVal) < 1e-10;
            bool isActual = Math.Abs(NumOps.ToDouble(actual[i]) - posVal) < 1e-10;
            if (isPred) pp++;
            if (isActual) ap++;
            if (isPred && isActual) tp++;
        }
        if (pp == 0 && ap == 0) return NumOps.One;
        double prec = pp > 0 ? (double)tp / pp : 0;
        double rec = ap > 0 ? (double)tp / ap : 0;
        if (prec + rec == 0) return NumOps.Zero;
        double b2 = _beta * _beta;
        return NumOps.FromDouble((1 + b2) * prec * rec / (b2 * prec + rec));
    }

    private T ComputeMicro(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual, HashSet<double> classes)
    {
        int totalTp = 0, totalPp = 0, totalAp = 0;
        foreach (var cls in classes)
        {
            for (int i = 0; i < pred.Length; i++)
            {
                bool isPred = Math.Abs(NumOps.ToDouble(pred[i]) - cls) < 1e-10;
                bool isActual = Math.Abs(NumOps.ToDouble(actual[i]) - cls) < 1e-10;
                if (isPred) totalPp++;
                if (isActual) totalAp++;
                if (isPred && isActual) totalTp++;
            }
        }
        double prec = totalPp > 0 ? (double)totalTp / totalPp : 0;
        double rec = totalAp > 0 ? (double)totalTp / totalAp : 0;
        if (prec + rec == 0) return NumOps.Zero;
        double b2 = _beta * _beta;
        return NumOps.FromDouble((1 + b2) * prec * rec / (b2 * prec + rec));
    }

    private static HashSet<double> GetClasses(ReadOnlySpan<T> vals)
    {
        var c = new HashSet<double>();
        for (int i = 0; i < vals.Length; i++) c.Add(NumOps.ToDouble(vals[i]));
        return c;
    }

    private static int CountClass(ReadOnlySpan<T> vals, double cls)
    {
        int c = 0;
        for (int i = 0; i < vals.Length; i++) if (Math.Abs(NumOps.ToDouble(vals[i]) - cls) < 1e-10) c++;
        return c;
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
