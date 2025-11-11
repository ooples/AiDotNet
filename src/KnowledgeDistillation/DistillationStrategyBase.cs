using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Abstract base class for knowledge distillation strategies.
/// Provides common functionality for computing losses and gradients in student-teacher training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TOutput">The output data type (typically Vector&lt;T&gt; or Matrix&lt;T&gt; of logits).</typeparam>
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
/// <para><b>Template Method Pattern:</b> The base class defines the structure (properties, validation),
/// and subclasses implement the specifics (loss computation logic).</para>
/// </remarks>
public abstract class DistillationStrategyBase<T, TOutput> : IDistillationStrategy<T, TOutput>
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
    /// Computes the distillation loss between student and teacher outputs.
    /// </summary>
    /// <param name="studentOutput">The student model's output (logits).</param>
    /// <param name="teacherOutput">The teacher model's output (logits).</param>
    /// <param name="trueLabels">Ground truth labels (optional).</param>
    /// <returns>The computed distillation loss value.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method to define your strategy's loss computation.
    /// The base class handles temperature and alpha; you focus on the loss calculation logic.</para>
    /// </remarks>
    public abstract T ComputeLoss(TOutput studentOutput, TOutput teacherOutput, TOutput? trueLabels = default);

    /// <summary>
    /// Computes the gradient of the distillation loss for backpropagation.
    /// </summary>
    /// <param name="studentOutput">The student model's output (logits).</param>
    /// <param name="teacherOutput">The teacher model's output (logits).</param>
    /// <param name="trueLabels">Ground truth labels (optional).</param>
    /// <returns>The gradient of the loss with respect to student outputs.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method to compute gradients for your strategy.
    /// The gradient should match the loss computation in ComputeLoss.</para>
    /// </remarks>
    public abstract TOutput ComputeGradient(TOutput studentOutput, TOutput teacherOutput, TOutput? trueLabels = default);

    /// <summary>
    /// Validates that student and teacher outputs have matching dimensions.
    /// </summary>
    /// <param name="studentOutput">Student output to validate.</param>
    /// <param name="teacherOutput">Teacher output to validate.</param>
    /// <param name="getDimension">Function to extract dimension from output.</param>
    /// <exception cref="ArgumentException">Thrown when dimensions don't match.</exception>
    protected void ValidateOutputDimensions(TOutput studentOutput, TOutput teacherOutput, Func<TOutput, int> getDimension)
    {
        if (studentOutput == null) throw new ArgumentNullException(nameof(studentOutput));
        if (teacherOutput == null) throw new ArgumentNullException(nameof(teacherOutput));

        int studentDim = getDimension(studentOutput);
        int teacherDim = getDimension(teacherOutput);

        if (studentDim != teacherDim)
        {
            throw new ArgumentException(
                $"Student and teacher output dimensions must match. Student: {studentDim}, Teacher: {teacherDim}");
        }
    }

    /// <summary>
    /// Validates that outputs and labels have matching dimensions (if labels are provided).
    /// </summary>
    /// <param name="output">Output to validate.</param>
    /// <param name="labels">Labels to validate (can be null).</param>
    /// <param name="getDimension">Function to extract dimension.</param>
    /// <exception cref="ArgumentException">Thrown when dimensions don't match.</exception>
    protected void ValidateLabelDimensions(TOutput output, TOutput? labels, Func<TOutput, int> getDimension)
    {
        if (labels == null) return;

        int outputDim = getDimension(output);
        int labelDim = getDimension(labels);

        if (outputDim != labelDim)
        {
            throw new ArgumentException(
                $"Output and label dimensions must match. Output: {outputDim}, Labels: {labelDim}");
        }
    }

    /// <summary>
    /// Gets the epsilon value for numerical stability (to avoid log(0), division by zero, etc.).
    /// </summary>
    protected const double Epsilon = 1e-10;
}
