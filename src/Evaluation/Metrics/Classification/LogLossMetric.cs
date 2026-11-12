using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Log Loss (Cross-Entropy Loss): a probabilistic measure that penalizes confident wrong predictions.
/// </summary>
/// <remarks>
/// <para>
/// For binary classification:
/// Log Loss = -1/N * Σ[y*log(p) + (1-y)*log(1-p)]
///
/// For multi-class:
/// Log Loss = -1/N * Σ Σ y_ik * log(p_ik)
/// </para>
/// <para>
/// <b>For Beginners:</b> Log loss measures how well predicted probabilities match actual outcomes.
/// Unlike accuracy which just checks if the prediction is correct, log loss considers the confidence:
/// <list type="bullet">
/// <item>Predicting 0.99 when correct: Low loss (good)</item>
/// <item>Predicting 0.51 when correct: Higher loss (less confident)</item>
/// <item>Predicting 0.99 when wrong: Very high loss (confidently wrong = bad)</item>
/// </list>
/// </para>
/// <para>
/// <b>Interpretation:</b>
/// <list type="bullet">
/// <item>Log Loss = 0: Perfect predictions with complete confidence</item>
/// <item>Lower is better</item>
/// <item>Log Loss approaches infinity for confidently wrong predictions</item>
/// </list>
/// </para>
/// <para>
/// <b>Use cases:</b> Essential for evaluating probabilistic classifiers, probability calibration,
/// and whenever predicted probabilities (not just class labels) matter.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LogLossMetric<T> : IProbabilisticClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _epsilon;

    /// <inheritdoc/>
    public string Name => "LogLoss";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Cross-entropy loss measuring quality of probabilistic predictions.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.LowerIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => default; // Unbounded (use default to represent no upper bound)

    /// <inheritdoc/>
    public bool RequiresProbabilities => true;

    /// <inheritdoc/>
    public bool SupportsMultiClass => true;

    /// <summary>
    /// Initializes a new Log Loss metric.
    /// </summary>
    /// <param name="epsilon">Small value to clip probabilities and avoid log(0). Default: 1e-15.</param>
    public LogLossMetric(double epsilon = 1e-15)
    {
        _epsilon = epsilon;
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses = 2)
    {
        if (probabilities.Length == 0)
        {
            return NumOps.Zero;
        }

        if (numClasses == 2)
        {
            // Binary classification: probabilities is P(class=1) for each sample
            if (probabilities.Length != actuals.Length)
            {
                throw new ArgumentException("Probabilities and actuals must have the same length for binary classification.");
            }

            return ComputeBinaryLogLoss(probabilities, actuals);
        }
        else
        {
            // Multi-class: probabilities is flattened (samples * classes)
            int numSamples = actuals.Length;
            if (probabilities.Length != numSamples * numClasses)
            {
                throw new ArgumentException($"Expected {numSamples * numClasses} probabilities for {numSamples} samples and {numClasses} classes.");
            }

            return ComputeMultiClassLogLoss(probabilities, actuals, numClasses);
        }
    }

    private T ComputeBinaryLogLoss(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var probs = EvaluationTensorOps<T>.ToTensor(probabilities);
        var y = EvaluationTensorOps<T>.ToTensor(actuals);

        // A NaN probability or label makes the loss NaN (Math.Log and the clip propagate it).
        if (EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.NanMask(probs)) > 0
            || EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.NanMask(y)) > 0)
        {
            return NumOps.FromDouble(double.NaN);
        }

        // Log loss = -[y*log(p) + (1-y)*log(1-p)] with p clipped to [eps, 1 - eps]. 1 - p is floored at eps
        // as well: in a type narrower than double, 1 - eps rounds to 1 and 1 - p would reach 0.
        var epsilon = NumOps.FromDouble(_epsilon);
        var p = engine.TensorClamp(probs, epsilon, NumOps.FromDouble(1 - _epsilon));
        var one = engine.TensorOnesLike(p);
        var logP = engine.TensorLog(p);
        var logOneMinusP = engine.TensorLog(engine.TensorMax(engine.TensorSubtract(one, p), epsilon));
        var perSample = engine.TensorAdd(
            engine.TensorMultiply(y, logP),
            engine.TensorMultiply(engine.TensorSubtract(one, y), logOneMinusP));

        return NumOps.Negate(engine.TensorMean(perSample));
    }

    private T ComputeMultiClassLogLoss(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        int numSamples = actuals.Length;
        var probMatrix = engine.Reshape(EvaluationTensorOps<T>.ToTensor(probabilities), new[] { numSamples, numClasses });

        // The probability of each sample's rounded true class; a sample whose class is out of range adds nothing.
        var oneHot = EvaluationTensorOps<T>.RoundedOneHot(EvaluationTensorOps<T>.ToTensor(actuals), numClasses);
        var trueClassProb = engine.ReduceSum(engine.TensorWhere(oneHot, probMatrix, engine.TensorZerosLike(probMatrix)), new[] { 1 }, false);
        var valid = engine.ReduceSum(oneHot, new[] { 1 }, false);

        if (EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.NanMask(trueClassProb)) > 0)
        {
            return NumOps.FromDouble(double.NaN);
        }

        var p = engine.TensorClamp(trueClassProb, NumOps.FromDouble(_epsilon), NumOps.FromDouble(1 - _epsilon));
        var loss = engine.TensorSum(engine.TensorMultiply(valid, engine.TensorLog(p)));

        return NumOps.Negate(NumOps.Divide(loss, NumOps.FromDouble(numSamples)));
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> probabilities,
        ReadOnlySpan<T> actuals,
        int numClasses = 2,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(probabilities, actuals, numClasses);
        var (lower, upper) = ComputeBootstrapCI(probabilities, actuals, numClasses, bootstrapSamples, confidenceLevel, randomSeed);

        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T lower, T upper) ComputeBootstrapCI(
        ReadOnlySpan<T> probabilities,
        ReadOnlySpan<T> actuals,
        int numClasses,
        int bootstrapSamples,
        double confidenceLevel,
        int? randomSeed)
    {
        int n = actuals.Length;
        if (n == 0)
        {
            return (NumOps.Zero, NumOps.FromDouble(10)); // Arbitrary upper bound
        }

        var random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        var bootstrapValues = new List<T>(bootstrapSamples);
        var probArray = probabilities.ToArray();
        var actualArray = actuals.ToArray();

        int probsPerSample = numClasses == 2 ? 1 : numClasses;

        for (int b = 0; b < bootstrapSamples; b++)
        {
            var sampledProb = new T[n * probsPerSample];
            var sampledActual = new T[n];

            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sampledActual[i] = actualArray[idx];

                for (int c = 0; c < probsPerSample; c++)
                {
                    sampledProb[i * probsPerSample + c] = probArray[idx * probsPerSample + c];
                }
            }

            bootstrapValues.Add(Compute(sampledProb, sampledActual, numClasses));
        }

        double alpha = 1 - confidenceLevel;
        int lowerIdx = (int)Math.Floor(alpha / 2 * bootstrapSamples);
        int upperIdx = (int)Math.Ceiling((1 - alpha / 2) * bootstrapSamples) - 1;

        lowerIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, lowerIdx));
        upperIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, upperIdx));

        return EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, lowerIdx, upperIdx);
    }
}
