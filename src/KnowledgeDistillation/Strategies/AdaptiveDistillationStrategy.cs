using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Adaptive distillation strategy that dynamically adjusts temperature based on student performance.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Adaptive distillation is like a tutor who adjusts their teaching
/// style based on how well the student is doing. If the student is struggling with a sample,
/// the strategy uses lower temperature (sharper focus on correct answer). If the student is
/// confident, it uses higher temperature (reveals more class relationships).</para>
///
/// <para><b>How It Works:</b>
/// - Monitor student's confidence/accuracy on each sample
/// - For hard samples (student struggles): Lower temperature, focus on getting it right
/// - For easy samples (student confident): Higher temperature, learn relationships
/// - Dynamically adjust per sample based on tracked performance</para>
///
/// <para><b>Real-world Analogy:</b>
/// A math tutor notices a student struggling with algebra but excelling at geometry.
/// The tutor spends more time on algebra fundamentals (lower temperature, focused)
/// while letting the student explore advanced geometry concepts (higher temperature, exploratory).</para>
///
/// <para><b>Benefits:</b>
/// - Curriculum learning: Automatically adjusts difficulty
/// - Efficient training: Focus effort where needed
/// - Better convergence: Avoids overwhelming student early
/// - Personalized: Adapts to this specific student's strengths/weaknesses</para>
///
/// <para><b>Architecture Note:</b> This strategy replaces the old AdaptiveTeacherModel approach.
/// Temperature adaptation belongs in the strategy layer (loss computation), not the teacher layer
/// (prediction generation). This follows Single Responsibility Principle.</para>
/// </remarks>
public class AdaptiveDistillationStrategy<T> : DistillationStrategyBase<T, Vector<T>>
{
    private readonly AdaptiveStrategy _strategy;
    private readonly double _minTemperature;
    private readonly double _maxTemperature;
    private readonly double _adaptationRate;
    private readonly Dictionary<int, double> _studentPerformance;

    /// <summary>
    /// Initializes a new instance of the AdaptiveDistillationStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="strategy">Adaptation strategy to use (default: ConfidenceBased).</param>
    /// <param name="minTemperature">Minimum temperature for hard samples (default: 1.0).</param>
    /// <param name="maxTemperature">Maximum temperature for easy samples (default: 5.0).</param>
    /// <param name="adaptationRate">How quickly to adapt, 0-1 (default: 0.1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create an adaptive strategy by providing:
    /// - baseTemperature: Starting point (3.0 typical)
    /// - strategy: How to measure difficulty (ConfidenceBased recommended)
    /// - min/maxTemperature: Range for adaptation (1.0-5.0 typical)</para>
    ///
    /// <para>Example:
    /// <code>
    /// var adaptiveStrategy = new AdaptiveDistillationStrategy&lt;double&gt;(
    ///     baseTemperature: 3.0,
    ///     alpha: 0.3,
    ///     strategy: AdaptiveStrategy.ConfidenceBased,
    ///     minTemperature: 1.0,  // Sharp for hard samples
    ///     maxTemperature: 5.0   // Soft for easy samples
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public AdaptiveDistillationStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        AdaptiveStrategy strategy = AdaptiveStrategy.ConfidenceBased,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
        : base(baseTemperature, alpha)
    {
        if (minTemperature <= 0 || maxTemperature <= minTemperature)
            throw new ArgumentException("Temperature range invalid: must have 0 < min < max");
        if (adaptationRate <= 0 || adaptationRate > 1)
            throw new ArgumentException("Adaptation rate must be in (0, 1]");

        _strategy = strategy;
        _minTemperature = minTemperature;
        _maxTemperature = maxTemperature;
        _adaptationRate = adaptationRate;
        _studentPerformance = new Dictionary<int, double>();
    }

    /// <summary>
    /// Updates student performance tracking for a specific sample.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <param name="studentOutput">Student's prediction.</param>
    /// <param name="trueLabel">True label (optional, needed for AccuracyBased strategy).</param>
    /// <remarks>
    /// <para>Call this after each forward pass to track student performance and enable
    /// adaptive temperature adjustment.</para>
    /// </remarks>
    public void UpdatePerformance(int sampleIndex, Vector<T> studentOutput, Vector<T>? trueLabel = null)
    {
        double performance = ComputePerformance(studentOutput, trueLabel);

        // Exponential moving average for smoothing
        if (_studentPerformance.ContainsKey(sampleIndex))
        {
            _studentPerformance[sampleIndex] =
                _adaptationRate * performance + (1 - _adaptationRate) * _studentPerformance[sampleIndex];
        }
        else
        {
            _studentPerformance[sampleIndex] = performance;
        }
    }

    /// <summary>
    /// Computes distillation loss with adaptive temperature based on sample difficulty.
    /// </summary>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);

        // Compute adaptive temperature for this sample
        double adaptiveTemp = ComputeAdaptiveTemperature(studentOutput, teacherOutput);

        // Compute soft loss with adaptive temperature
        var studentSoft = Softmax(studentOutput, adaptiveTemp);
        var teacherSoft = Softmax(teacherOutput, adaptiveTemp);

        var softLoss = KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(adaptiveTemp * adaptiveTemp));

        // Add hard loss if labels provided
        if (trueLabels != null)
        {
            var studentProbs = Softmax(studentOutput, temperature: 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);

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

        // Compute adaptive temperature
        double adaptiveTemp = ComputeAdaptiveTemperature(studentOutput, teacherOutput);

        // Soft gradient with adaptive temperature
        var studentSoft = Softmax(studentOutput, adaptiveTemp);
        var teacherSoft = Softmax(teacherOutput, adaptiveTemp);

        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(adaptiveTemp * adaptiveTemp));
        }

        // Add hard gradient if labels provided
        if (trueLabels != null)
        {
            var studentProbs = Softmax(studentOutput, temperature: 1.0);

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

    private double ComputePerformance(Vector<T> studentOutput, Vector<T>? trueLabel)
    {
        switch (_strategy)
        {
            case AdaptiveStrategy.ConfidenceBased:
                // Max confidence as performance measure
                return GetMaxConfidence(studentOutput);

            case AdaptiveStrategy.AccuracyBased:
                if (trueLabel != null)
                {
                    // Correctness as performance measure
                    return IsCorrect(studentOutput, trueLabel) ? 1.0 : 0.0;
                }
                return 0.5; // Default if no label

            case AdaptiveStrategy.EntropyBased:
                // Low entropy = confident (high performance)
                var probs = Softmax(studentOutput, 1.0);
                return 1.0 - ComputeEntropy(probs);

            default:
                return 0.5;
        }
    }

    private double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        double difficulty;

        switch (_strategy)
        {
            case AdaptiveStrategy.ConfidenceBased:
                // Lower confidence = harder sample = lower temperature
                var probs = Softmax(studentOutput, 1.0);
                difficulty = 1.0 - GetMaxConfidence(probs);
                break;

            case AdaptiveStrategy.EntropyBased:
                // Higher entropy = harder sample = lower temperature
                var probsEnt = Softmax(studentOutput, 1.0);
                difficulty = ComputeEntropy(probsEnt);
                break;

            case AdaptiveStrategy.AccuracyBased:
            default:
                // Medium difficulty by default
                difficulty = 0.5;
                break;
        }

        // Clamp difficulty to [0, 1]
        difficulty = Math.Max(0.0, Math.Min(1.0, difficulty));

        // Map difficulty to temperature range
        // High difficulty (hard sample) -> low temperature (sharper, focused)
        // Low difficulty (easy sample) -> high temperature (softer, exploratory)
        return _minTemperature + (1.0 - difficulty) * (_maxTemperature - _minTemperature);
    }

    private double GetMaxConfidence(Vector<T> probs)
    {
        T maxVal = probs[0];
        for (int i = 1; i < probs.Length; i++)
        {
            if (NumOps.GreaterThan(probs[i], maxVal))
                maxVal = probs[i];
        }
        return Convert.ToDouble(maxVal);
    }

    private double ComputeEntropy(Vector<T> probs)
    {
        double entropy = 0;
        const double epsilon = 1e-10;

        for (int i = 0; i < probs.Length; i++)
        {
            double p = Convert.ToDouble(probs[i]);
            if (p > epsilon)
            {
                entropy -= p * Math.Log(p);
            }
        }

        // Normalize by max entropy
        double maxEntropy = Math.Log(probs.Length);
        return maxEntropy > 0 ? entropy / maxEntropy : 0;
    }

    private bool IsCorrect(Vector<T> prediction, Vector<T> label)
    {
        int predClass = ArgMax(prediction);
        int trueClass = ArgMax(label);
        return predClass == trueClass;
    }

    private int ArgMax(Vector<T> vector)
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
}

/// <summary>
/// Defines how the adaptive strategy measures sample difficulty.
/// </summary>
public enum AdaptiveStrategy
{
    /// <summary>
    /// Adjust based on student's prediction confidence (max probability).
    /// Lower confidence = harder sample.
    /// </summary>
    ConfidenceBased,

    /// <summary>
    /// Adjust based on student's correctness (requires labels).
    /// Incorrect predictions = harder sample.
    /// </summary>
    AccuracyBased,

    /// <summary>
    /// Adjust based on student's prediction entropy (uncertainty).
    /// Higher entropy = more uncertain = harder sample.
    /// </summary>
    EntropyBased
}
