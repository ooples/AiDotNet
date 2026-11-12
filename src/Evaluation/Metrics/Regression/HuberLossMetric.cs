using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Huber Loss: a robust loss function that is less sensitive to outliers than MSE.
/// </summary>
/// <remarks>
/// <para>
/// Huber Loss = (1/N) * Σ L_δ(y, ŷ) where:
/// <list type="bullet">
/// <item>L_δ = 0.5 * (y - ŷ)² if |y - ŷ| ≤ δ (quadratic for small errors)</item>
/// <item>L_δ = δ * (|y - ŷ| - 0.5 * δ) if |y - ŷ| > δ (linear for large errors)</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Huber loss combines the best of MSE and MAE:
/// <list type="bullet">
/// <item>For small errors (≤ δ): Uses squared error like MSE (smooth gradient)</item>
/// <item>For large errors (> δ): Uses linear error like MAE (robust to outliers)</item>
/// </list>
/// The delta parameter controls where the transition happens. Common values are 1.0 or 1.35.</para>
/// </remarks>
public class HuberLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _delta;

    /// <inheritdoc />
    public string Name => "HuberLoss";
    /// <inheritdoc />
    public string Category => "Regression";
    /// <inheritdoc />
    public string Description => "Robust loss function combining MSE and MAE.";
    /// <inheritdoc />
    public MetricDirection Direction => MetricDirection.LowerIsBetter;
    /// <inheritdoc />
    public T? MinValue => NumOps.Zero;
    /// <inheritdoc />
    public T? MaxValue => default;
    /// <inheritdoc />
    public bool RequiresProbabilities => false;
    /// <inheritdoc />
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Initializes the Huber Loss metric.
    /// </summary>
    /// <param name="delta">Threshold at which to switch from quadratic to linear. Default is 1.0.</param>
    public HuberLossMetric(double delta = 1.0)
    {
        _delta = Math.Max(0.001, delta);
    }

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var delta = NumOps.FromDouble(_delta);
        var error = engine.TensorAbs(engine.TensorSubtract(actual, pred));

        // Quadratic region 0.5 * e^2 where e <= delta, linear region delta * (e - 0.5 * delta) elsewhere (NaN included)
        var quadratic = engine.TensorMultiplyScalar(engine.TensorSquare(error), NumOps.FromDouble(0.5));
        var linear = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(error, NumOps.FromDouble(0.5 * _delta)), delta);
        var inQuadratic = engine.TensorLessEqual(error, engine.TensorFullLike(error, delta));

        return engine.TensorMean(engine.TensorWhere(inQuadratic, quadratic, linear));
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
        if (n == 0) return (NumOps.Zero, NumOps.Zero);
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
