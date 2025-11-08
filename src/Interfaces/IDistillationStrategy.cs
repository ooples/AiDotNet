namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a strategy for computing knowledge distillation loss between student and teacher models.
/// </summary>
/// <typeparam name="TOutput">The output data type (typically Vector or Matrix of logits).</typeparam>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A distillation strategy determines how to measure the difference
/// between what the student predicts and what the teacher predicts. Different strategies can
/// focus on different aspects:
/// - Response-based: Compare final outputs
/// - Feature-based: Compare intermediate layer features
/// - Relation-based: Compare relationships between samples</para>
///
/// <para>The most common approach (Hinton et al., 2015) combines two losses:
/// 1. Hard loss: How well the student matches the true labels
/// 2. Soft loss: How well the student mimics the teacher's predictions</para>
///
/// <para>This combination allows the student to both get the right answers (hard loss) and
/// learn the teacher's reasoning (soft loss).</para>
/// </remarks>
public interface IDistillationStrategy<in TOutput, T>
{
    /// <summary>
    /// Computes the distillation loss between student and teacher outputs.
    /// </summary>
    /// <param name="studentOutput">The student model's output (logits).</param>
    /// <param name="teacherOutput">The teacher model's output (logits).</param>
    /// <param name="trueLabels">Ground truth labels (optional, can be null for unsupervised distillation).</param>
    /// <returns>The computed distillation loss value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how different the student's predictions are
    /// from the teacher's predictions. A lower loss means the student is learning well from the teacher.</para>
    ///
    /// <para>The formula typically used is:
    /// Total Loss = α × Hard Loss + (1 - α) × Soft Loss
    ///
    /// Where:
    /// - Hard Loss: Cross-entropy between student predictions and true labels
    /// - Soft Loss: KL divergence between student and teacher (with temperature scaling)
    /// - α (alpha): Balance parameter (typically 0.3-0.5)</para>
    /// </remarks>
    T ComputeLoss(TOutput studentOutput, TOutput teacherOutput, TOutput? trueLabels = default);

    /// <summary>
    /// Computes the gradient of the distillation loss for backpropagation.
    /// </summary>
    /// <param name="studentOutput">The student model's output (logits).</param>
    /// <param name="teacherOutput">The teacher model's output (logits).</param>
    /// <param name="trueLabels">Ground truth labels (optional).</param>
    /// <returns>The gradient of the loss with respect to student outputs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients tell us how to adjust the student model's parameters
    /// to reduce the loss. They point in the direction of steepest increase in loss, so we move
    /// in the opposite direction during training.</para>
    ///
    /// <para>The gradient combines information from both the teacher (soft targets) and the
    /// true labels (hard targets), helping the student learn from both sources.</para>
    /// </remarks>
    TOutput ComputeGradient(TOutput studentOutput, TOutput teacherOutput, TOutput? trueLabels = default);

    /// <summary>
    /// Gets or sets the temperature parameter for softening probability distributions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls how "soft" the predictions become:
    /// - T = 1: Normal predictions (standard softmax)
    /// - T = 2-10: Softer predictions that reveal more about class relationships
    /// - Higher T: Even softer, but gradients become smaller</para>
    ///
    /// <para>Typical values: 3-5 for most applications, 2-3 for easier tasks, 5-10 for harder tasks.</para>
    /// </remarks>
    double Temperature { get; set; }

    /// <summary>
    /// Gets or sets the balance parameter (alpha) between hard loss and soft loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alpha controls the trade-off between learning from true labels
    /// and learning from the teacher:
    /// - α = 0: Only learn from teacher (pure distillation)
    /// - α = 0.3-0.5: Balanced (recommended for most cases)
    /// - α = 1: Only learn from true labels (standard training, no distillation)</para>
    ///
    /// <para>When true labels are noisy or scarce, lower alpha (more weight on teacher) helps.
    /// When labels are clean and abundant, higher alpha (more weight on labels) works better.</para>
    /// </remarks>
    double Alpha { get; set; }
}
