using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Abstract base class for curriculum distillation strategies with progressive difficulty adjustment.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality for curriculum learning,
/// including progress tracking, sample difficulty management, and temperature progression.</para>
///
/// <para><b>For Implementers:</b> Derive from this class and implement
/// <see cref="ComputeCurriculumTemperature"/> and <see cref="ShouldIncludeSample"/> to define
/// your specific curriculum progression logic.</para>
///
/// <para><b>Shared Features:</b>
/// - Curriculum progress tracking (0.0 to 1.0)
/// - Sample difficulty scoring and management
/// - Temperature range validation
/// - Step/epoch-based progression</para>
/// </remarks>
public abstract class CurriculumDistillationStrategyBase<T>
    : DistillationStrategyBase<T>, ICurriculumDistillationStrategy<T>
{
    private readonly Dictionary<int, double> _sampleDifficulties;
    private int _currentStep;

    /// <summary>
    /// Gets the total number of steps in the curriculum.
    /// </summary>
    public int TotalSteps { get; }

    /// <summary>
    /// Gets the current curriculum progress (0.0 to 1.0).
    /// </summary>
    public double CurriculumProgress => (double)_currentStep / TotalSteps;

    /// <summary>
    /// Gets the minimum temperature for the curriculum.
    /// </summary>
    public double MinTemperature { get; }

    /// <summary>
    /// Gets the maximum temperature for the curriculum.
    /// </summary>
    public double MaxTemperature { get; }

    /// <summary>
    /// Initializes a new instance of the CurriculumDistillationStrategyBase class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Minimum temperature for curriculum (default: 2.0).</param>
    /// <param name="maxTemperature">Maximum temperature for curriculum (default: 5.0).</param>
    /// <param name="totalSteps">Total training steps/epochs (default: 100).</param>
    /// <param name="sampleDifficulties">Optional pre-defined difficulty scores.</param>
    protected CurriculumDistillationStrategyBase(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 2.0,
        double maxTemperature = 5.0,
        int totalSteps = 100,
        Dictionary<int, double>? sampleDifficulties = null)
        : base(baseTemperature, alpha)
    {
        if (minTemperature <= 0 || maxTemperature <= minTemperature)
            throw new ArgumentException("Temperature range invalid: must have 0 < min < max");
        if (totalSteps <= 0)
            throw new ArgumentException("Total steps must be positive", nameof(totalSteps));

        MinTemperature = minTemperature;
        MaxTemperature = maxTemperature;
        TotalSteps = totalSteps;
        _sampleDifficulties = sampleDifficulties ?? new Dictionary<int, double>();
        _currentStep = 0;
    }

    /// <summary>
    /// Updates the current curriculum progress.
    /// </summary>
    public virtual void UpdateProgress(int step)
    {
        _currentStep = Math.Max(0, Math.Min(step, TotalSteps - 1));
    }

    /// <summary>
    /// Sets the difficulty score for a specific sample.
    /// </summary>
    public virtual void SetSampleDifficulty(int sampleIndex, double difficulty)
    {
        difficulty = Math.Max(0.0, Math.Min(1.0, difficulty));
        _sampleDifficulties[sampleIndex] = difficulty;
    }

    /// <summary>
    /// Gets the difficulty score for a sample, if set.
    /// </summary>
    public virtual double? GetSampleDifficulty(int sampleIndex)
    {
        return _sampleDifficulties.TryGetValue(sampleIndex, out double difficulty) ? difficulty : null;
    }

    /// <summary>
    /// Determines if a sample should be included in current curriculum stage.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override to define strategy-specific sample filtering logic.</para>
    /// <para>Default: Includes all samples (no filtering).</para>
    /// </remarks>
    public abstract bool ShouldIncludeSample(int sampleIndex);

    /// <summary>
    /// Computes the curriculum-adjusted temperature based on current progress.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override to define strategy-specific temperature progression.</para>
    /// </remarks>
    public abstract double ComputeCurriculumTemperature();

    /// <summary>
    /// Computes distillation loss with curriculum-adjusted temperature.
    /// </summary>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Compute curriculum-adjusted temperature
        double curriculumTemp = ComputeCurriculumTemperature();

        // Soft loss per sample with curriculum temperature: T² KL(teacher || student)
        var studentSoft = DistillationHelper<T>.SoftmaxRows(studentBatchOutput, curriculumTemp);
        var teacherSoft = DistillationHelper<T>.SoftmaxRows(teacherBatchOutput, curriculumTemp);
        Vector<T> sampleLosses = Engine.Multiply(
            DistillationHelper<T>.KLDivergenceRows(teacherSoft, studentSoft),
            NumOps.FromDouble(curriculumTemp * curriculumTemp));

        // Add hard loss if labels provided: α × hard + (1 - α) × soft
        if (trueLabelsBatch != null)
        {
            var studentProbs = DistillationHelper<T>.SoftmaxRows(studentBatchOutput, temperature: 1.0);
            var hardLosses = DistillationHelper<T>.CrossEntropyRows(studentProbs, trueLabelsBatch);

            var alphaT = NumOps.FromDouble(Alpha);
            var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

            sampleLosses = Engine.Add(
                Engine.Multiply(hardLosses, alphaT),
                Engine.Multiply(sampleLosses, oneMinusAlpha));
        }

        T totalLoss = Engine.Sum(sampleLosses);
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient with curriculum-adjusted temperature.
    /// </summary>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Compute curriculum-adjusted temperature
        double curriculumTemp = ComputeCurriculumTemperature();

        // Soft gradient with curriculum temperature: (student_soft - teacher_soft) × T
        var studentSoft = DistillationHelper<T>.SoftmaxRows(studentBatchOutput, curriculumTemp);
        var teacherSoft = DistillationHelper<T>.SoftmaxRows(teacherBatchOutput, curriculumTemp);
        var gradientBatch = Engine.MatrixMultiplyScalar(
            Engine.MatrixSubtract(studentSoft, teacherSoft), NumOps.FromDouble(curriculumTemp));

        // Add hard gradient if labels provided: α × (probs - labels) + (1 - α) × soft
        if (trueLabelsBatch != null)
        {
            var studentProbs = DistillationHelper<T>.SoftmaxRows(studentBatchOutput, temperature: 1.0);
            gradientBatch = Engine.MatrixAdd(
                Engine.MatrixMultiplyScalar(Engine.MatrixSubtract(studentProbs, trueLabelsBatch), NumOps.FromDouble(Alpha)),
                Engine.MatrixMultiplyScalar(gradientBatch, NumOps.FromDouble(1.0 - Alpha)));
        }

        // Average gradients over the batch
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        gradientBatch = Engine.MatrixMultiplyScalar(gradientBatch, oneOverBatch);

        return gradientBatch;
    }

    /// <summary>
    /// Clamps a value to the temperature range [MinTemperature, MaxTemperature].
    /// </summary>
    protected double ClampTemperature(double temperature)
    {
        return Math.Max(MinTemperature, Math.Min(MaxTemperature, temperature));
    }
}
