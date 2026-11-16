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

    /// <inheritdoc />
    public string Name => "Specificity";
    /// <inheritdoc />
    public string Category => "Classification";
    /// <inheritdoc />
    public string Description => "Proportion of actual negatives correctly identified (TN / (TN + FP)).";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => false;

    public SpecificityMetric()
    {
        _positiveLabel = NumOps.One;
    }

    public SpecificityMetric(T positiveLabel)
    {
        _positiveLabel = positiveLabel;
    }

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");

        if (predictions.Length == 0) return NumOps.Zero;

        var (_, trueNegatives, fp, _) = EvaluationTensorOps<T>.BinaryConfusion(predictions, actuals, _positiveLabel);
        int actualNegatives = trueNegatives + fp;

        return actualNegatives > 0 ? NumOps.FromDouble((double)trueNegatives / actualNegatives) : NumOps.One;
    }

    /// <inheritdoc />
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
        var values = new List<T>(samples);
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();

        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = predArr[idx]; sa[i] = actArr[idx]; }
            values.Add(Compute(sp, sa));
        }
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return EvaluationTensorOps<T>.OrderStatistics(values, lo, hi);
    }
}
