using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Tweedie Deviance Loss for regression with power parameter.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Tweedie loss is flexible for different data types:
/// <list type="bullet">
/// <item>Power = 0: Normal distribution (like MSE)</item>
/// <item>Power = 1: Poisson distribution (count data)</item>
/// <item>Power = 2: Gamma distribution (positive continuous)</item>
/// <item>Power = 3: Inverse Gaussian distribution</item>
/// <item>1 &lt; Power &lt; 2: Compound Poisson-Gamma (insurance claims)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Insurance claim prediction (zeros and positives)</item>
/// <item>Sales forecasting with many zeros</item>
/// <item>Any data with mixed zero and positive values</item>
/// </list>
/// </para>
/// </remarks>
public class TweedieLossMetric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double _power;

    public TweedieLossMetric(double power = 1.5)
    {
        if (power < 0 || (power > 0 && power < 1))
            throw new ArgumentException("Power must be 0 or >= 1.");
        _power = power;
    }

    /// <inheritdoc />
    public string Name => $"TweedieLoss(p={_power})";
    /// <inheritdoc />
    public string Category => "Regression";
    /// <inheritdoc />
    public string Description => $"Tweedie Deviance Loss with power parameter {_power}.";
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

    /// <inheritdoc />
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var y = EvaluationTensorOps<T>.ToTensor(actuals).Cast<double>();
        var mu = EvaluationTensorOps<double>.MaxPreservingNaN(
            EvaluationTensorOps<T>.ToTensor(predictions).Cast<double>(), 1e-10);
        var zeros = engine.TensorZerosLike(y);
        var yPositive = engine.TensorGreaterThan(y, 0.0);
        Tensor<double> loss;

        if (Math.Abs(_power) < 1e-10)
        {
            loss = engine.TensorSquare(engine.TensorSubtract(y, mu));
        }
        else if (Math.Abs(_power - 1) < 1e-10)
        {
            var positive = engine.TensorMultiplyScalar(
                engine.TensorSubtract(engine.TensorMultiply(y, engine.TensorLog(engine.TensorDivide(y, mu))), engine.TensorSubtract(y, mu)),
                2.0);
            loss = engine.TensorWhere(yPositive, positive, engine.TensorMultiplyScalar(mu, 2.0));
        }
        else if (Math.Abs(_power - 2) < 1e-10)
        {
            var positive = engine.TensorMultiplyScalar(
                engine.TensorSubtract(engine.TensorDivide(engine.TensorSubtract(y, mu), mu), engine.TensorLog(engine.TensorDivide(y, mu))),
                2.0);
            loss = engine.TensorWhere(yPositive, positive, zeros);
        }
        else
        {
            var a = engine.TensorMultiplyScalar(
                EvaluationTensorOps<double>.MaxPreservingNaN(engine.TensorPower(y, 2 - _power), 1e-10),
                1.0 / ((1 - _power) * (2 - _power)));
            var b = engine.TensorMultiplyScalar(
                engine.TensorMultiply(y, engine.TensorPower(mu, 1 - _power)), 1.0 / (1 - _power));
            var c = engine.TensorMultiplyScalar(engine.TensorPower(mu, 2 - _power), 1.0 / (2 - _power));
            loss = engine.TensorMultiplyScalar(engine.TensorAdd(engine.TensorSubtract(a, b), c), 2.0);
        }
        var (totalLoss, validCount) = EvaluationTensorOps<double>.FiniteSumAndCount(loss);
        return validCount > 0 ? NumOps.FromDouble(totalLoss / validCount) : NumOps.Zero;
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
