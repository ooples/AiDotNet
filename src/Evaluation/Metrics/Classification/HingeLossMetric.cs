using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Hinge Loss: loss function used by Support Vector Machines (SVM).
/// </summary>
/// <remarks>
/// <para>Hinge Loss = (1/N) * Σmax(0, 1 - y_i * ŷ_i) where y ∈ {-1, +1}</para>
/// <para><b>For Beginners:</b> Hinge loss penalizes predictions that are on the wrong side
/// of the decision boundary OR are correct but not confident enough.
/// <list type="bullet">
/// <item>Loss = 0: All predictions are correct with margin ≥ 1</item>
/// <item>Loss increases when predictions are wrong or uncertain</item>
/// </list>
/// Used in SVM training and for evaluating margin-based classifiers.</para>
/// <para><b>Important:</b> Predictions should be raw decision function values (continuous,
/// typically unbounded) from SVM, not class labels. Actuals should be class labels (0/1 or -1/+1).
/// If predictions are already binary class labels, use 0-1 loss instead.</para>
/// </remarks>
public class HingeLossMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "HingeLoss";
    /// <inheritdoc />
    public string Category => "Classification";
    /// <inheritdoc />
    public string Description => "SVM margin-based loss function.";
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
        var pred = EvaluationTensorOps<T>.ToTensor(predictions);
        var actual = EvaluationTensorOps<T>.ToTensor(actuals);
        var one = engine.TensorOnesLike(pred);

        // Predictions should be raw decision function values (continuous scores); actuals are class labels
        // (0/1 or -1/+1). A 0/1-encoded actual maps to -1/+1: y = actual <= 0.5 ? -1 : 1.
        var negative = engine.TensorLessEqual(actual, engine.TensorFullLike(actual, NumOps.FromDouble(0.5)));
        var y = engine.TensorSubtract(one, engine.TensorMultiplyScalar(negative, NumOps.FromDouble(2.0)));

        // Predictions in [0, 1] are likely probabilities and are moved to signed margin space (2 * pred - 1);
        // other scores are used as they are.
        var inUnitRange = engine.TensorMultiply(
            engine.TensorGreaterEqual(pred, engine.TensorZerosLike(pred)),
            engine.TensorLessEqual(pred, one));
        var yHat = engine.TensorAdd(pred, engine.TensorMultiply(inUnitRange, engine.TensorSubtract(pred, one)));

        // Hinge loss: max(0, 1 - y * yHat). Math.Max propagates NaN, the engine max does not.
        var margin = engine.TensorSubtract(one, engine.TensorMultiply(y, yHat));
        if (EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.NanMask(margin)) > 0) return NumOps.FromDouble(double.NaN);

        return engine.TensorMean(engine.TensorMax(margin, NumOps.Zero));
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
