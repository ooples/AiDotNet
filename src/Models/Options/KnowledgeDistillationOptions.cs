using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for knowledge distillation training.
/// </summary>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class configures how knowledge distillation works.
/// Think of it as the "settings" for transferring knowledge from a large teacher model
/// to a smaller student model.</para>
///
/// <para><b>Quick Start Example:</b>
/// <code>
/// var options = new KnowledgeDistillationOptions&lt;Vector&lt;double&gt;, Vector&lt;double&gt;, double&gt;
/// {
///     TeacherModelType = TeacherModelType.NeuralNetwork,
///     StrategyType = DistillationStrategyType.ResponseBased,
///     Temperature = 3.0,  // Soft predictions
///     Alpha = 0.3,        // 30% hard labels, 70% teacher
///     Epochs = 20,
///     BatchSize = 32
/// };
/// </code>
/// </para>
/// </remarks>
public class KnowledgeDistillationOptions<TInput, TOutput, T>
{
    /// <summary>
    /// Gets or sets the type of teacher model to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The teacher is the "expert" model. Choose:
    /// - NeuralNetwork: Standard pre-trained model
    /// - Ensemble: Multiple teachers for better knowledge
    /// - Self: Model teaches itself (no separate teacher needed)</para>
    /// </remarks>
    public TeacherModelType TeacherModelType { get; set; } = TeacherModelType.NeuralNetwork;

    /// <summary>
    /// Gets or sets the distillation strategy type.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The strategy determines what knowledge to transfer:
    /// - ResponseBased: Match final outputs (most common)
    /// - FeatureBased: Match intermediate layers
    /// - AttentionBased: Match attention patterns (for transformers)</para>
    /// </remarks>
    public DistillationStrategyType StrategyType { get; set; } = DistillationStrategyType.ResponseBased;

    /// <summary>
    /// Gets or sets the teacher model instance (if using pre-instantiated teacher).
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Provide a custom teacher model instance.
    /// If null, one will be created based on TeacherModelType.</para>
    /// </remarks>
    public ITeacherModel<TInput, TOutput>? Teacher { get; set; }

    /// <summary>
    /// Gets or sets the distillation strategy instance (if using custom strategy).
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Provide a custom distillation strategy.
    /// If null, one will be created based on StrategyType.</para>
    /// </remarks>
    public IDistillationStrategy<TOutput, T>? Strategy { get; set; }

    /// <summary>
    /// Gets or sets the temperature for softmax scaling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls how "soft" predictions are:
    /// - Low (1-2): Sharp predictions
    /// - Medium (3-5): Balanced (recommended)
    /// - High (6-10): Very soft predictions</para>
    ///
    /// <para>Higher temperature reveals more about class relationships but may be harder to optimize.</para>
    /// </remarks>
    public double Temperature { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the alpha parameter balancing hard and soft loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alpha controls the balance:
    /// - 0.0: Only learn from teacher
    /// - 0.3-0.5: Balanced (recommended)
    /// - 1.0: Only learn from labels (no distillation)</para>
    ///
    /// <para>Use lower alpha when labels are noisy or you want to rely more on the teacher.</para>
    /// </remarks>
    public double Alpha { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> An epoch is one complete pass through the training data.
    /// Typical values: 10-50 epochs depending on dataset size and complexity.</para>
    /// </remarks>
    public int Epochs { get; set; } = 20;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batch size is how many samples to process at once:
    /// - Small (16-32): Less memory, noisier gradients
    /// - Medium (64-128): Balanced
    /// - Large (256+): More memory, smoother gradients</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the learning rate for student training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Learning rate controls how fast the student learns:
    /// - Too low: Slow training
    /// - Too high: Unstable training
    /// - Typical: 0.001-0.01</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets whether to use label smoothing.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Label smoothing softens hard labels slightly,
    /// which can improve generalization. Usually not needed with distillation.</para>
    /// </remarks>
    public bool UseLabelSmoothing { get; set; } = false;

    /// <summary>
    /// Gets or sets the label smoothing factor (if enabled).
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 0.1-0.2. Higher values smooth labels more.</para>
    /// </remarks>
    public double LabelSmoothingFactor { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to freeze teacher model during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Usually true - teacher should remain fixed.
    /// Set to false for online distillation where teacher updates with student.</para>
    /// </remarks>
    public bool FreezeTeacher { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set a seed to get reproducible results.
    /// Useful for debugging and comparing experiments.</para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets callback function invoked after each epoch.
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Use this to log progress, save checkpoints,
    /// or implement custom logic during training.</para>
    /// </remarks>
    public Action<int, T>? OnEpochComplete { get; set; }

    /// <summary>
    /// Gets or sets whether to validate model after each epoch.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, evaluates student on validation set
    /// after each epoch to monitor progress.</para>
    /// </remarks>
    public bool ValidateAfterEpoch { get; set; } = true;

    /// <summary>
    /// Gets or sets validation data inputs (if validation is enabled).
    /// </summary>
    public TInput[]? ValidationInputs { get; set; }

    /// <summary>
    /// Gets or sets validation data labels (if validation is enabled).
    /// </summary>
    public TOutput[]? ValidationLabels { get; set; }

    /// <summary>
    /// Gets or sets layer pairs for feature-based distillation.
    /// Format: "teacher_layer:student_layer"
    /// </summary>
    /// <remarks>
    /// <para><b>For Feature-Based Distillation:</b> Specify which layers to match.
    /// Example: ["conv3:conv2", "conv4:conv3"]</para>
    /// </remarks>
    public string[]? FeatureLayerPairs { get; set; }

    /// <summary>
    /// Gets or sets weight for feature loss (if using feature-based distillation).
    /// </summary>
    /// <remarks>
    /// <para>Controls how much to weight feature matching vs output matching.
    /// Typical values: 0.3-0.7</para>
    /// </remarks>
    public double FeatureWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets attention layer names (if using attention-based distillation).
    /// </summary>
    /// <remarks>
    /// <para><b>For Attention-Based Distillation:</b> Specify attention layers to match.
    /// Example: ["attention1", "attention2"]</para>
    /// </remarks>
    public string[]? AttentionLayers { get; set; }

    /// <summary>
    /// Gets or sets weight for attention loss (if using attention-based distillation).
    /// </summary>
    /// <remarks>
    /// <para>Controls how much to weight attention matching.
    /// Typical values: 0.2-0.4</para>
    /// </remarks>
    public double AttentionWeight { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to use exponential moving average for teacher predictions (self-distillation).
    /// </summary>
    /// <remarks>
    /// <para><b>For Self-Distillation:</b> EMA smooths teacher predictions over time,
    /// improving stability.</para>
    /// </remarks>
    public bool UseEMA { get; set; } = false;

    /// <summary>
    /// Gets or sets the EMA decay rate (if using EMA).
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 0.99-0.999. Higher values give more weight to history.</para>
    /// </remarks>
    public double EMADecay { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the number of self-distillation generations (if using self-distillation).
    /// </summary>
    /// <remarks>
    /// <para><b>For Self-Distillation:</b> How many times the model re-teaches itself.
    /// Typical values: 1-3 generations.</para>
    /// </remarks>
    public int SelfDistillationGenerations { get; set; } = 1;

    /// <summary>
    /// Validates the options and throws if any are invalid.
    /// </summary>
    public void Validate()
    {
        if (Temperature <= 0)
            throw new ArgumentException("Temperature must be positive", nameof(Temperature));
        if (Alpha < 0 || Alpha > 1)
            throw new ArgumentException("Alpha must be between 0 and 1", nameof(Alpha));
        if (Epochs <= 0)
            throw new ArgumentException("Epochs must be positive", nameof(Epochs));
        if (BatchSize <= 0)
            throw new ArgumentException("BatchSize must be positive", nameof(BatchSize));
        if (LearningRate <= 0)
            throw new ArgumentException("LearningRate must be positive", nameof(LearningRate));
        if (FeatureWeight < 0 || FeatureWeight > 1)
            throw new ArgumentException("FeatureWeight must be between 0 and 1", nameof(FeatureWeight));
        if (AttentionWeight < 0 || AttentionWeight > 1)
            throw new ArgumentException("AttentionWeight must be between 0 and 1", nameof(AttentionWeight));
        if (EMADecay < 0 || EMADecay > 1)
            throw new ArgumentException("EMADecay must be between 0 and 1", nameof(EMADecay));
        if (SelfDistillationGenerations < 1)
            throw new ArgumentException("SelfDistillationGenerations must be at least 1", nameof(SelfDistillationGenerations));

        if (ValidateAfterEpoch && (ValidationInputs == null || ValidationLabels == null))
            throw new ArgumentException("Validation data must be provided when ValidateAfterEpoch is true");
    }
}
