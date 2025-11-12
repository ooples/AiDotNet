using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Abstract base class for adaptive distillation strategies with performance tracking.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality for all adaptive
/// strategies, including performance tracking with exponential moving average and temperature
/// range management.</para>
///
/// <para><b>For Implementers:</b> Derive from this class and implement
/// <see cref="ComputeAdaptiveTemperature"/> to define your specific adaptation logic.</para>
///
/// <para><b>Shared Features:</b>
/// - Exponential moving average (EMA) for performance tracking
/// - Temperature range validation and enforcement
/// - Performance history management
/// - Helper methods for confidence, entropy, and accuracy calculations</para>
/// </remarks>
public abstract class AdaptiveDistillationStrategyBase<T>
    : DistillationStrategyBase<T, Vector<T>>, IAdaptiveDistillationStrategy<T>
{
    private readonly Dictionary<int, double> _studentPerformance;

    /// <summary>
    /// Gets the minimum temperature for adaptation.
    /// </summary>
    public double MinTemperature { get; }

    /// <summary>
    /// Gets the maximum temperature for adaptation.
    /// </summary>
    public double MaxTemperature { get; }

    /// <summary>
    /// Gets the adaptation rate for exponential moving average.
    /// </summary>
    public double AdaptationRate { get; }

    /// <summary>
    /// Initializes a new instance of the AdaptiveDistillationStrategyBase class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Minimum temperature for adaptation (default: 1.0).</param>
    /// <param name="maxTemperature">Maximum temperature for adaptation (default: 5.0).</param>
    /// <param name="adaptationRate">Rate for EMA performance tracking (default: 0.1).</param>
    protected AdaptiveDistillationStrategyBase(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
        : base(baseTemperature, alpha)
    {
        if (minTemperature <= 0)
            throw new ArgumentException("Minimum temperature must be positive", nameof(minTemperature));
        if (maxTemperature <= minTemperature)
            throw new ArgumentException("Maximum temperature must be greater than minimum", nameof(maxTemperature));
        if (adaptationRate <= 0 || adaptationRate > 1)
            throw new ArgumentException("Adaptation rate must be in (0, 1]", nameof(adaptationRate));

        MinTemperature = minTemperature;
        MaxTemperature = maxTemperature;
        AdaptationRate = adaptationRate;
        _studentPerformance = new Dictionary<int, double>();
    }

    /// <summary>
    /// Updates the performance metric for a specific sample using exponential moving average.
    /// </summary>
    public virtual void UpdatePerformance(int sampleIndex, Vector<T> studentOutput, Vector<T>? trueLabel = null)
    {
        double performance = ComputePerformance(studentOutput, trueLabel);

        // Exponential moving average: new = α * current + (1-α) * old
        if (_studentPerformance.ContainsKey(sampleIndex))
        {
            _studentPerformance[sampleIndex] =
                AdaptationRate * performance + (1 - AdaptationRate) * _studentPerformance[sampleIndex];
        }
        else
        {
            _studentPerformance[sampleIndex] = performance;
        }
    }

    /// <summary>
    /// Gets the current performance metric for a sample.
    /// </summary>
    public virtual double GetPerformance(int sampleIndex)
    {
        return _studentPerformance.TryGetValue(sampleIndex, out double perf) ? perf : 0.5;
    }

    /// <summary>
    /// Computes the adaptive temperature for a specific sample.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this to define strategy-specific temperature adaptation.</para>
    /// </remarks>
    public abstract double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput);

    /// <summary>
    /// Computes distillation loss with adaptive temperature.
    /// </summary>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);

        // Compute adaptive temperature for this sample
        double adaptiveTemp = ComputeAdaptiveTemperature(studentOutput, teacherOutput);

        // Compute soft loss with adaptive temperature
        var studentSoft = DistillationHelper<T>.Softmax(studentOutput, adaptiveTemp);
        var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, adaptiveTemp);

        var softLoss = DistillationHelper<T>.KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(adaptiveTemp * adaptiveTemp));

        // Add hard loss if labels provided
        if (trueLabels != null)
        {
            var studentProbs = DistillationHelper<T>.Softmax(studentOutput, temperature: 1.0);
            var hardLoss = DistillationHelper<T>.CrossEntropy(studentProbs, trueLabels);

            var alphaT = NumOps.FromDouble(Alpha);
            var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

            return NumOps.Add(
                NumOps.Multiply(alphaT, hardLoss),
                NumOps.Multiply(oneMinusAlpha, softLoss));
        }

        return softLoss;
    }

    /// <summary>
    /// Computes gradient with adaptive temperature.
    /// </summary>
    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);

        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);

        // Compute adaptive temperature for this sample
        double adaptiveTemp = ComputeAdaptiveTemperature(studentOutput, teacherOutput);

        // Soft gradient with adaptive temperature
        var studentSoft = DistillationHelper<T>.Softmax(studentOutput, adaptiveTemp);
        var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, adaptiveTemp);

        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(adaptiveTemp * adaptiveTemp));
        }

        // Add hard gradient if labels provided
        if (trueLabels != null)
        {
            var studentProbs = DistillationHelper<T>.Softmax(studentOutput, temperature: 1.0);

            for (int i = 0; i < n; i++)
            {
                var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);
                var alphaWeighted = NumOps.Multiply(hardGrad, NumOps.FromDouble(Alpha));
                var softWeighted = NumOps.Multiply(gradient[i], NumOps.FromDouble(1.0 - Alpha));
                gradient[i] = NumOps.Add(alphaWeighted, softWeighted);
            }
        }
        else
        {
            // Scale by (1 - alpha) if no hard loss
            for (int i = 0; i < n; i++)
            {
                gradient[i] = NumOps.Multiply(gradient[i], NumOps.FromDouble(1.0 - Alpha));
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes a performance metric for the student output.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override to define strategy-specific performance metrics.</para>
    /// <para>Default: Returns max confidence (highest probability).</para>
    /// </remarks>
    protected virtual double ComputePerformance(Vector<T> studentOutput, Vector<T>? trueLabel)
    {
        var probs = DistillationHelper<T>.Softmax(studentOutput, 1.0);
        return GetMaxConfidence(probs);
    }

    /// <summary>
    /// Gets the maximum confidence (highest probability) from a probability distribution.
    /// </summary>
    protected double GetMaxConfidence(Vector<T> probabilities)
    {
        double max = 0.0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            double prob = Convert.ToDouble(probabilities[i]);
            if (prob > max) max = prob;
        }
        return max;
    }

    /// <summary>
    /// Computes the entropy of a probability distribution.
    /// </summary>
    /// <remarks>
    /// <para>Entropy measures uncertainty. Higher entropy = more uncertain = harder sample.</para>
    /// </remarks>
    protected double ComputeEntropy(Vector<T> probabilities)
    {
        double entropy = 0.0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            double prob = Convert.ToDouble(probabilities[i]);
            if (prob > 1e-10) // Avoid log(0)
            {
                entropy -= prob * Math.Log(prob);
            }
        }
        // Normalize to [0, 1] range
        return entropy / Math.Log(probabilities.Length);
    }

    /// <summary>
    /// Checks if the student prediction is correct.
    /// </summary>
    protected bool IsCorrect(Vector<T> studentOutput, Vector<T> trueLabel)
    {
        return ArgMax(studentOutput) == ArgMax(trueLabel);
    }

    /// <summary>
    /// Finds the index of the maximum value in a vector.
    /// </summary>
    protected int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Clamps a value to the temperature range [MinTemperature, MaxTemperature].
    /// </summary>
    protected double ClampTemperature(double temperature)
    {
        return Math.Max(MinTemperature, Math.Min(MaxTemperature, temperature));
    }
}
