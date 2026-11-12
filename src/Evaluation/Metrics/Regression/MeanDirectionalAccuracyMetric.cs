using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Mean Directional Accuracy (MDA): fraction of correctly predicted directions.
/// </summary>
/// <remarks>
/// <para>MDA = (1/N) × Σ I(sign(y_t - y_{t-1}) = sign(ŷ_t - ŷ_{t-1}))</para>
/// <para><b>For Beginners:</b> MDA measures if you predicted the right direction:
/// <list type="bullet">
/// <item>Did the model predict "up" when actual went up?</item>
/// <item>Range: 0 to 1, higher is better</item>
/// <item>0.5 = random guessing</item>
/// <item>Important in trading/forecasting where direction matters more than magnitude</item>
/// </list>
/// </para>
/// </remarks>
public class MeanDirectionalAccuracyMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "MDA";
    /// <inheritdoc />
    public string Category => "Regression";
    /// <inheritdoc />
    public string Description => "Mean Directional Accuracy (fraction of correct direction predictions).";
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

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length < 2) return NumOps.FromDouble(0.5);

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var actualChange = engine.TensorDiff(actual, 1, 0);
        var predChange = engine.TensorDiff(pred, 1, 0);
        var zeros = engine.TensorZerosLike(actualChange);
        var tolerance = engine.TensorFullLike(actualChange, NumOps.FromDouble(1e-10));

        // Directions match when both rise, both fall, or both stay (within 1e-10)
        var bothUp = engine.TensorMultiply(engine.TensorGreaterThan(actualChange, zeros), engine.TensorGreaterThan(predChange, zeros));
        var bothDown = engine.TensorMultiply(engine.TensorLessThan(actualChange, zeros), engine.TensorLessThan(predChange, zeros));
        var bothFlat = engine.TensorMultiply(
            engine.TensorLessThan(engine.TensorAbs(actualChange), tolerance),
            engine.TensorLessThan(engine.TensorAbs(predChange), tolerance));
        var match = engine.TensorMax(engine.TensorMax(bothUp, bothDown), bothFlat);

        int correct = EvaluationTensorOps<T>.Count(match);
        int total = actualChange.Length;
        return total > 0 ? NumOps.FromDouble((double)correct / total) : NumOps.FromDouble(0.5);
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
        if (n < 2) return (NumOps.Zero, NumOps.One);
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
