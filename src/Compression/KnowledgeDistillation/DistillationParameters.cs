namespace AiDotNet.Compression.KnowledgeDistillation;

/// <summary>
/// Parameters for knowledge distillation.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates parameters that control the knowledge distillation process.
/// </para>
/// <para><b>For Beginners:</b> These are the settings that control how distillation works.
/// 
/// These parameters determine:
/// - How the teacher's outputs are processed
/// - How learning balances between soft and hard targets
/// - How long training continues
/// - Which specific distillation method is used
/// </para>
/// </remarks>
public class DistillationParameters
{
    /// <summary>
    /// Gets or sets the temperature for softening logits.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The temperature parameter controls how "soft" the teacher model's probability
    /// distribution is. Higher values make the distribution more uniform.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how smooth the teacher's outputs are.
    /// 
    /// Temperature works by dividing the logits (pre-softmax values):
    /// - Higher temperature (>1) makes probabilities more uniform
    /// - Lower temperature (<1) makes probabilities more peaked
    /// - Typical values are 2.0-5.0 for distillation
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the weight between soft and hard targets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter (0-1) determines the balance between learning from soft targets
    /// and learning from hard targets. A value of 1.0 means using only soft targets.
    /// </para>
    /// <para><b>For Beginners:</b> This balances learning from the teacher vs. true labels.
    /// 
    /// For example:
    /// - Alpha = 0.7 means 70% of learning comes from the teacher's outputs
    /// - The remaining 30% comes from the true labels
    /// - Setting this properly helps the student learn effectively
    /// </para>
    /// </remarks>
    public double Alpha { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets a value indicating whether to use soft targets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the teacher's probability distributions are used as soft targets.
    /// When false, only the teacher's predicted classes are used.
    /// </para>
    /// <para><b>For Beginners:</b> This determines if the student learns from probabilities or just answers.
    /// 
    /// For example:
    /// - True: Student learns from full probability distributions (like 0.7, 0.2, 0.1)
    /// - False: Student learns only from final predictions (like "cat", "dog")
    /// - Using soft targets (True) generally leads to better distillation
    /// </para>
    /// </remarks>
    public bool UseSoftTargets { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of complete passes through the training data during distillation.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many times the student sees the training data.
    /// 
    /// More epochs:
    /// - Give the student more opportunities to learn
    /// - May lead to better performance
    /// - But increase training time
    /// - And may lead to overfitting if too many
    /// </para>
    /// </remarks>
    public int TrainingEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the distillation method to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The specific knowledge distillation method to apply during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is the specific technique used for distillation.
    /// 
    /// Different methods have different approaches:
    /// - Vanilla uses the original distillation approach
    /// - Attention also distills attention patterns
    /// - Feature uses intermediate layer outputs
    /// 
    /// These methods involve different ways of transferring knowledge.
    /// </para>
    /// </remarks>
    public DistillationMethod Method { get; set; } = DistillationMethod.Vanilla;
}