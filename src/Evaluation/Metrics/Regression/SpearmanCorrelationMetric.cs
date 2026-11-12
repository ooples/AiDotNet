using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Spearman's Rank Correlation Coefficient between predictions and actuals.
/// </summary>
/// <remarks>
/// <para>ρ = 1 - 6Σd²/(n(n²-1)) where d = rank difference</para>
/// <para><b>For Beginners:</b> Spearman correlation measures monotonic relationships:
/// <list type="bullet">
/// <item>Range: -1 to 1</item>
/// <item>1 = perfect positive monotonic relationship</item>
/// <item>-1 = perfect negative monotonic relationship</item>
/// <item>Non-parametric (uses ranks, not values)</item>
/// <item>Robust to outliers unlike Pearson correlation</item>
/// </list>
/// </para>
/// </remarks>
public class SpearmanCorrelationMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "SpearmanCorrelation";
    /// <inheritdoc />
    public string Category => "Regression";
    /// <inheritdoc />
    public string Description => "Spearman's Rank Correlation Coefficient.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.FromDouble(-1.0);
    /// <inheritdoc />
    public T? MaxValue => NumOps.One;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => false;

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length < 2) return NumOps.Zero;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();

        // Compute ranks (tied values share their mean rank)
        var predRanks = EvaluationTensorOps<T>.MidRanks(EvaluationTensorOps<T>.ToTensor(predictions));
        var actualRanks = EvaluationTensorOps<T>.MidRanks(EvaluationTensorOps<T>.ToTensor(actuals));

        // Compute Spearman correlation using Pearson correlation on ranks
        // This correctly handles ties (the d² formula assumes no ties)
        var dp = engine.TensorSubtractScalar(predRanks, engine.TensorMean(predRanks));
        var da = engine.TensorSubtractScalar(actualRanks, engine.TensorMean(actualRanks));
        T cov = engine.TensorSum(engine.TensorMultiply(dp, da));
        T varPred = engine.TensorSumOfSquares(dp);
        T varActual = engine.TensorSumOfSquares(da);

        // Guard against zero variance (constant ranks)
        var tolerance = NumOps.FromDouble(1e-12);
        if (NumOps.LessThan(varPred, tolerance) || NumOps.LessThan(varActual, tolerance))
            return NumOps.Zero;

        return NumOps.Divide(cov, NumOps.Sqrt(NumOps.Multiply(varPred, varActual)));
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
        if (n < 2) return (NumOps.FromDouble(-1), NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var values = new List<T>(samples);
        var predArr = pred.ToArray(); var actArr = actual.ToArray();
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
