using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Curriculum distillation strategy that progressively adjusts training difficulty.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Curriculum learning is like starting with easy problems
/// and gradually increasing difficulty as the student improves. This strategy adjusts
/// the distillation process based on training progress.</para>
///
/// <para><b>How It Works:</b>
/// - Track training progress (epochs, steps, or custom metrics)
/// - Optionally score samples by difficulty
/// - Adjust temperature based on curriculum stage
/// - Easy-to-Hard: Start with high temperature (soft targets), lower over time (sharper)
/// - Hard-to-Easy: Start with low temperature (sharp targets), increase over time (softer)</para>
///
/// <para><b>Real-world Analogy:</b>
/// Teaching calculus: Start with basic derivatives (easy), gradually introduce
/// integration (medium), then complex differential equations (hard). The curriculum
/// ensures students build strong foundations before tackling advanced topics.</para>
///
/// <para><b>Benefits:</b>
/// - Stable training: Easier samples help student learn fundamentals
/// - Better generalization: Progressive difficulty prevents overfitting
/// - Faster convergence: Student not overwhelmed early in training
/// - Flexible: Can use epoch-based or sample-difficulty-based curriculum</para>
///
/// <para><b>Architecture Note:</b> This strategy replaces the old CurriculumTeacherModel approach.
/// Curriculum logic belongs in the strategy layer (training process), not the teacher layer
/// (prediction generation). This follows Single Responsibility Principle.</para>
/// </remarks>
public class CurriculumDistillationStrategy<T> : DistillationStrategyBase<T, Vector<T>>
{
    private readonly CurriculumStrategy _strategy;
    private readonly double _minTemperature;
    private readonly double _maxTemperature;
    private readonly int _totalSteps;
    private readonly Dictionary<int, double>? _sampleDifficulties;
    private int _currentStep;

    /// <summary>
    /// Initializes a new instance of the CurriculumDistillationStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="strategy">Curriculum strategy direction (default: EasyToHard).</param>
    /// <param name="minTemperature">Starting temperature for easy samples (default: 2.0).</param>
    /// <param name="maxTemperature">Ending temperature for hard samples (default: 5.0).</param>
    /// <param name="totalSteps">Total training steps/epochs for curriculum (default: 100).</param>
    /// <param name="sampleDifficulties">Optional difficulty scores per sample index (0.0=easy, 1.0=hard).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create a curriculum strategy by providing:
    /// - strategy: Direction of curriculum (EasyToHard recommended)
    /// - totalSteps: How many epochs/steps the curriculum spans
    /// - min/maxTemperature: Temperature range for progression
    /// - sampleDifficulties: Optional per-sample difficulty scores</para>
    ///
    /// <para>Example:
    /// <code>
    /// var curriculumStrategy = new CurriculumDistillationStrategy&lt;double&gt;(
    ///     baseTemperature: 3.0,
    ///     alpha: 0.3,
    ///     strategy: CurriculumStrategy.EasyToHard,
    ///     minTemperature: 2.0,  // Start easy (higher temp)
    ///     maxTemperature: 5.0,  // End hard (lower temp)
    ///     totalSteps: 100       // 100 epochs
    /// );
    ///
    /// // Update progress each epoch
    /// for (int epoch = 0; epoch &lt; 100; epoch++)
    /// {
    ///     curriculumStrategy.UpdateProgress(epoch);
    ///     // ... training loop
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public CurriculumDistillationStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        CurriculumStrategy strategy = CurriculumStrategy.EasyToHard,
        double minTemperature = 2.0,
        double maxTemperature = 5.0,
        int totalSteps = 100,
        Dictionary<int, double>? sampleDifficulties = null)
        : base(baseTemperature, alpha)
    {
        if (minTemperature <= 0 || maxTemperature <= minTemperature)
            throw new ArgumentException("Temperature range invalid: must have 0 < min < max");
        if (totalSteps <= 0)
            throw new ArgumentException("Total steps must be positive");

        _strategy = strategy;
        _minTemperature = minTemperature;
        _maxTemperature = maxTemperature;
        _totalSteps = totalSteps;
        _sampleDifficulties = sampleDifficulties;
        _currentStep = 0;
    }

    /// <summary>
    /// Updates the current curriculum progress.
    /// </summary>
    /// <param name="step">Current training step/epoch (0 to totalSteps-1).</param>
    /// <remarks>
    /// <para>Call this at the beginning of each epoch or training step to update
    /// the curriculum progress and adjust temperature accordingly.</para>
    /// </remarks>
    public void UpdateProgress(int step)
    {
        _currentStep = Math.Max(0, Math.Min(step, _totalSteps - 1));
    }

    /// <summary>
    /// Sets difficulty scores for specific samples.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <param name="difficulty">Difficulty score (0.0 = easy, 1.0 = hard).</param>
    /// <remarks>
    /// <para>Use this to manually set difficulty scores if not provided at construction.
    /// Difficulty scores are used to determine sample order in curriculum.</para>
    /// </remarks>
    public void SetSampleDifficulty(int sampleIndex, double difficulty)
    {
        if (_sampleDifficulties == null)
            throw new InvalidOperationException("Sample difficulties not initialized. Provide dictionary in constructor.");

        difficulty = Math.Max(0.0, Math.Min(1.0, difficulty));
        _sampleDifficulties[sampleIndex] = difficulty;
    }

    /// <summary>
    /// Gets the current curriculum progress as a ratio (0.0 to 1.0).
    /// </summary>
    public double CurriculumProgress => (double)_currentStep / _totalSteps;

    /// <summary>
    /// Computes distillation loss with curriculum-adjusted temperature.
    /// </summary>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);

        // Compute curriculum-adjusted temperature
        double curriculumTemp = ComputeCurriculumTemperature();

        // Compute soft loss with curriculum temperature
        var studentSoft = Softmax(studentOutput, curriculumTemp);
        var teacherSoft = Softmax(teacherOutput, curriculumTemp);

        var softLoss = KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(curriculumTemp * curriculumTemp));

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
    /// Computes gradient with curriculum-adjusted temperature.
    /// </summary>
    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);

        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);

        // Compute curriculum-adjusted temperature
        double curriculumTemp = ComputeCurriculumTemperature();

        // Soft gradient with curriculum temperature
        var studentSoft = Softmax(studentOutput, curriculumTemp);
        var teacherSoft = Softmax(teacherOutput, curriculumTemp);

        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(curriculumTemp * curriculumTemp));
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

    /// <summary>
    /// Determines if a sample should be included in current curriculum stage.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <returns>True if sample should be included in current training.</returns>
    /// <remarks>
    /// <para>Use this to filter training samples based on curriculum progress.
    /// Only samples within the current difficulty range are included.</para>
    /// </remarks>
    public bool ShouldIncludeSample(int sampleIndex)
    {
        if (_sampleDifficulties == null || !_sampleDifficulties.ContainsKey(sampleIndex))
            return true; // Include all samples if no difficulty info

        double difficulty = _sampleDifficulties[sampleIndex];
        double progress = CurriculumProgress;

        switch (_strategy)
        {
            case CurriculumStrategy.EasyToHard:
                // Include samples with difficulty <= current progress
                return difficulty <= progress;

            case CurriculumStrategy.HardToEasy:
                // Include samples with difficulty >= (1 - current progress)
                return difficulty >= (1.0 - progress);

            default:
                return true;
        }
    }

    private double ComputeCurriculumTemperature()
    {
        double progress = CurriculumProgress;

        switch (_strategy)
        {
            case CurriculumStrategy.EasyToHard:
                // Start with max temperature (easy/soft), decrease to min (hard/sharp)
                // High temperature early = softer targets (easier to learn)
                // Low temperature late = sharper targets (more challenging)
                return _maxTemperature - progress * (_maxTemperature - _minTemperature);

            case CurriculumStrategy.HardToEasy:
                // Start with min temperature (hard/sharp), increase to max (easy/soft)
                // Low temperature early = sharper targets (challenging)
                // High temperature late = softer targets (easier)
                return _minTemperature + progress * (_maxTemperature - _minTemperature);

            default:
                return BaseTemperature;
        }
    }
}

/// <summary>
/// Defines the curriculum learning strategy direction.
/// </summary>
/// <remarks>
/// <para>This enum is used by CurriculumDistillationStrategy to control
/// the progression of training difficulty over time.</para>
/// </remarks>
public enum CurriculumStrategy
{
    /// <summary>
    /// Start with easy examples and gradually increase difficulty.
    /// Recommended for most scenarios - builds strong foundations.
    /// </summary>
    EasyToHard,

    /// <summary>
    /// Start with hard examples and gradually decrease difficulty.
    /// Useful for fine-tuning or when student has prior knowledge.
    /// </summary>
    HardToEasy
}
