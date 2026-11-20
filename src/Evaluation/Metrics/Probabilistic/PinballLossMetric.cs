using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Probabilistic;

/// <summary>
/// Pinball loss (quantile loss) metric for probabilistic predictions.
/// </summary>
/// <remarks>
/// <para>
/// Pinball loss is the standard loss function for quantile regression.
/// It asymmetrically penalizes over and under-predictions based on the quantile level.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you're predicting quantiles (like the median, or 90th percentile),
/// pinball loss tells you how good your predictions are. It penalizes under-predictions and
/// over-predictions differently based on which quantile you're targeting.
///
/// For the median (τ=0.5), under and over-predictions are penalized equally.
/// For the 90th percentile (τ=0.9), under-predictions are penalized more heavily.
/// </para>
/// <para>
/// Loss = τ * max(y - ŷ, 0) + (1 - τ) * max(ŷ - y, 0)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class PinballLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _tau;

    /// <summary>
    /// Initializes a new Pinball Loss metric for the specified quantile.
    /// </summary>
    /// <param name="tau">The quantile level (must be in (0, 1)). Default is 0.5 (median).</param>
    public PinballLossMetric(double tau = 0.5)
    {
        if (tau <= 0 || tau >= 1)
            throw new ArgumentOutOfRangeException(nameof(tau), "Quantile must be in (0, 1).");

        _tau = tau;
    }

    /// <inheritdoc/>
    public string Name => $"PinballLoss_{_tau:F2}";

    /// <inheritdoc/>
    public string Category => "Probabilistic";

    /// <inheritdoc/>
    public string Description => $"Quantile loss for τ={_tau:F2} - measures accuracy of quantile predictions.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.LowerIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => default;

    /// <inheritdoc/>
    public bool RequiresProbabilities => false;

    /// <inheritdoc/>
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Gets the quantile level.
    /// </summary>
    public double Tau => _tau;

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var pred = EvaluationTensorOps<T>.ToTensor(predictions).Cast<double>();
        var actual = EvaluationTensorOps<T>.ToTensor(actuals).Cast<double>();
        var diff = engine.TensorSubtract(actual, pred);
        var over = engine.TensorMultiplyScalar(diff, _tau);
        var under = engine.TensorMultiplyScalar(engine.TensorNegate(diff), 1 - _tau);
        var nonNegative = engine.TensorGreaterEqual(diff, engine.TensorZerosLike(diff));
        return NumOps.FromDouble(engine.TensorMean(engine.TensorWhere(nonNegative, over, under)));
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
    {
        if (ciMethod != ConfidenceIntervalMethod.PercentileBootstrap)
        {
            throw new NotSupportedException(
                $"PinballLossMetric only supports PercentileBootstrap. '{ciMethod}' is not supported.");
        }
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual,
        int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.Zero);

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var predArr = pred.ToArray();
        var actArr = actual.ToArray();

        // Generate bootstrap samples (reuse buffers to reduce GC pressure)
        var bootstrapValues = new List<T>(samples);
        var sp = new T[n];
        var sa = new T[n];
        for (int b = 0; b < samples; b++)
        {
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sp[i] = predArr[idx];
                sa[i] = actArr[idx];
            }
            bootstrapValues.Add(Compute(sp, sa));
        }

        double alpha = 1 - conf;
        var bootstrapTensor = new Tensor<T>(bootstrapValues.ToArray(), new[] { samples });

        // ComputeWithCI admits only PercentileBootstrap.
        return PercentileInterval(bootstrapTensor, alpha);
    }

    private static (T, T) PercentileInterval(Tensor<T> bootstrapValues, double alpha)
    {
        int count = bootstrapValues.Length;
        int lo = Math.Max(0, (int)(alpha / 2 * count));
        int hi = Math.Min(count - 1, (int)((1 - alpha / 2) * count) - 1);
        hi = Math.Max(hi, lo + 1);
        hi = Math.Min(hi, count - 1);
        return EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, lo, hi);
    }
}
