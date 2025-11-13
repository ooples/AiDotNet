using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Abstract base class for knowledge distillation strategies.
/// Provides common functionality for computing losses and gradients in student-teacher training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A distillation strategy defines how to measure the difference
/// between student and teacher predictions. This base class provides common functionality that
/// all strategies need, like temperature and alpha parameters.</para>
///
/// <para><b>Design Philosophy:</b>
/// Different distillation strategies focus on different aspects:
/// - **Response-based**: Match final outputs (logits/probabilities)
/// - **Feature-based**: Match intermediate layer representations
/// - **Relation-based**: Match relationships between samples
/// - **Attention-based**: Match attention patterns (for transformers)</para>
///
/// <para>This base class ensures all strategies handle temperature and alpha consistently,
/// while allowing flexibility in how loss is computed.</para>
///
/// <para><b>Batch Processing:</b> All strategies now operate on batches (Matrix&lt;T&gt;) for efficiency.
/// Each row in the matrices represents one sample in the batch.</para>
///
/// <para><b>Template Method Pattern:</b> The base class defines the structure (properties, validation),
/// and subclasses implement the specifics (loss computation logic).</para>
/// </remarks>
public abstract class DistillationStrategyBase<T> : IDistillationStrategy<T>
{
    /// <summary>
    /// Numeric operations for the specified type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    private double _temperature;
    private double _alpha;

    /// <summary>
    /// Gets or sets the temperature parameter for softening probability distributions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher temperature makes predictions "softer", revealing
    /// more about the model's uncertainty and class relationships.</para>
    ///
    /// <para>Temperature effects:
    /// - T = 1: Standard predictions (sharp)
    /// - T = 2-5: Softer predictions (recommended for distillation)
    /// - T > 10: Very soft (may be too smooth)</para>
    ///
    /// <para><b>Validation:</b> Must be positive (&gt; 0). Setting invalid values throws ArgumentException.</para>
    /// </remarks>
    public double Temperature
    {
        get => _temperature;
        set
        {
            if (value <= 0)
                throw new ArgumentException("Temperature must be positive (> 0)", nameof(value));
            _temperature = value;
        }
    }

    /// <summary>
    /// Gets or sets the balance parameter between hard loss and soft loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alpha controls the trade-off between learning from true labels
    /// (hard loss) and learning from the teacher (soft loss).</para>
    ///
    /// <para>Alpha values:
    /// - α = 0: Only learn from teacher (pure distillation)
    /// - α = 0.3-0.5: Balanced (recommended)
    /// - α = 1: Only learn from labels (no distillation)</para>
    ///
    /// <para><b>Validation:</b> Must be between 0 and 1. Setting invalid values throws ArgumentException.</para>
    /// </remarks>
    public double Alpha
    {
        get => _alpha;
        set
        {
            if (value < 0 || value > 1)
                throw new ArgumentException("Alpha must be between 0 and 1 (inclusive)", nameof(value));
            _alpha = value;
        }
    }

    /// <summary>
    /// Initializes a new instance of the distillation strategy base class.
    /// </summary>
    /// <param name="temperature">Softmax temperature (default 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default 0.3).</param>
    protected DistillationStrategyBase(double temperature = 3.0, double alpha = 0.3)
    {
        // Validate and set through properties (which have validation logic)
        Temperature = temperature;
        Alpha = alpha;
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes the distillation loss between student and teacher batch outputs.
    /// </summary>
    /// <param name="studentBatchOutput">The student model's output logits for a batch. Shape: [batch_size x num_classes]</param>
    /// <param name="teacherBatchOutput">The teacher model's output logits for a batch. Shape: [batch_size x num_classes]</param>
    /// <param name="trueLabelsBatch">Ground truth labels for the batch (optional). Shape: [batch_size x num_classes]</param>
    /// <returns>The computed distillation loss value (scalar) for the batch.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method to define your strategy's loss computation.
    /// The base class handles temperature and alpha; you focus on the loss calculation logic.</para>
    ///
    /// <para><b>Batch Processing:</b> The loss should be computed over all samples in the batch
    /// and typically averaged. Each row in the input matrices represents one sample.</para>
    /// </remarks>
    public abstract T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null);

    /// <summary>
    /// Computes the gradient of the distillation loss for backpropagation.
    /// </summary>
    /// <param name="studentBatchOutput">The student model's output logits for a batch. Shape: [batch_size x num_classes]</param>
    /// <param name="teacherBatchOutput">The teacher model's output logits for a batch. Shape: [batch_size x num_classes]</param>
    /// <param name="trueLabelsBatch">Ground truth labels for the batch (optional). Shape: [batch_size x num_classes]</param>
    /// <returns>The gradient of the loss with respect to student outputs. Shape: [batch_size x num_classes]</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method to compute gradients for your strategy.
    /// The gradient should match the loss computation in ComputeLoss.</para>
    ///
    /// <para><b>Batch Processing:</b> Returns a gradient matrix with the same shape as the input,
    /// one gradient row for each sample in the batch.</para>
    /// </remarks>
    public abstract Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null);

    /// <summary>
    /// Validates that student and teacher batch outputs have matching dimensions.
    /// </summary>
    /// <param name="studentBatchOutput">Student batch output to validate.</param>
    /// <param name="teacherBatchOutput">Teacher batch output to validate.</param>
    /// <exception cref="ArgumentNullException">Thrown when outputs are null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions don't match.</exception>
    /// <remarks>
    /// <para>Checks both batch size (rows) and output dimension (columns) match between student and teacher.</para>
    /// </remarks>
    protected void ValidateOutputDimensions(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput)
    {
        if (studentBatchOutput == null) throw new ArgumentNullException(nameof(studentBatchOutput));
        if (teacherBatchOutput == null) throw new ArgumentNullException(nameof(teacherBatchOutput));

        if (studentBatchOutput.RowCount != teacherBatchOutput.RowCount)
        {
            throw new ArgumentException(
                $"Student and teacher batch sizes must match. Student: {studentBatchOutput.RowCount}, Teacher: {teacherBatchOutput.RowCount}");
        }

        if (studentBatchOutput.ColumnCount != teacherBatchOutput.ColumnCount)
        {
            throw new ArgumentException(
                $"Student and teacher output dimensions must match. Student: {studentBatchOutput.ColumnCount}, Teacher: {teacherBatchOutput.ColumnCount}");
        }
    }

    /// <summary>
    /// Validates that batch outputs and labels have matching dimensions (if labels are provided).
    /// </summary>
    /// <param name="batchOutput">Batch output to validate.</param>
    /// <param name="labelsBatch">Labels batch to validate (can be null).</param>
    /// <exception cref="ArgumentException">Thrown when dimensions don't match.</exception>
    /// <remarks>
    /// <para>If labels are null, validation is skipped (for pure soft distillation without labels).</para>
    /// </remarks>
    protected void ValidateLabelDimensions(Matrix<T> batchOutput, Matrix<T>? labelsBatch)
    {
        if (labelsBatch == null) return;

        if (batchOutput.RowCount != labelsBatch.RowCount)
        {
            throw new ArgumentException(
                $"Output and label batch sizes must match. Output: {batchOutput.RowCount}, Labels: {labelsBatch.RowCount}");
        }

        if (batchOutput.ColumnCount != labelsBatch.ColumnCount)
        {
            throw new ArgumentException(
                $"Output and label dimensions must match. Output: {batchOutput.ColumnCount}, Labels: {labelsBatch.ColumnCount}");
        }
    }

    /// <summary>
    /// Gets the epsilon value for numerical stability (to avoid log(0), division by zero, etc.).
    /// </summary>
    protected const double Epsilon = 1e-10;
}
