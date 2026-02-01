using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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

    private static double Clamp(double value, double min, double max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
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
        int n = probabilities.Length;
        double loss = 0;

        for (int i = 0; i < n; i++)
        {
            double p = Clamp(NumOps.ToDouble(probabilities[i]), _epsilon, 1 - _epsilon);
            double y = NumOps.ToDouble(actuals[i]);

            // Log loss = -[y*log(p) + (1-y)*log(1-p)]
            loss -= y * Math.Log(p) + (1 - y) * Math.Log(1 - p);
        }

        return NumOps.FromDouble(loss / n);
    }

    private T ComputeMultiClassLogLoss(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses)
    {
        int numSamples = actuals.Length;
        double loss = 0;

        for (int i = 0; i < numSamples; i++)
        {
            int trueClass = (int)Math.Round(NumOps.ToDouble(actuals[i]));

            if (trueClass >= 0 && trueClass < numClasses)
            {
                double p = Clamp(NumOps.ToDouble(probabilities[i * numClasses + trueClass]), _epsilon, 1 - _epsilon);
                loss -= Math.Log(p);
            }
        }

        return NumOps.FromDouble(loss / numSamples);
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> probabilities,
        ReadOnlySpan<T> actuals,
        int numClasses = 2,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.BCaBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
    {
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
            : new Random();

        var bootstrapValues = new double[bootstrapSamples];
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

            bootstrapValues[b] = NumOps.ToDouble(Compute(sampledProb, sampledActual, numClasses));
        }

        Array.Sort(bootstrapValues);

        double alpha = 1 - confidenceLevel;
        int lowerIdx = (int)Math.Floor(alpha / 2 * bootstrapSamples);
        int upperIdx = (int)Math.Ceiling((1 - alpha / 2) * bootstrapSamples) - 1;

        lowerIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, lowerIdx));
        upperIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, upperIdx));

        return (NumOps.FromDouble(bootstrapValues[lowerIdx]), NumOps.FromDouble(bootstrapValues[upperIdx]));
    }
}
