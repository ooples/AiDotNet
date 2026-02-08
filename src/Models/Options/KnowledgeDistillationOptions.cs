using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for knowledge distillation training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class configures how knowledge distillation works.
/// Think of it as the "settings" for transferring knowledge from a large teacher model
/// to a smaller student model.</para>
///
/// <para><b>Quick Start Example:</b>
/// <code>
/// var options = new KnowledgeDistillationOptions&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;
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
public class KnowledgeDistillationOptions<T, TInput, TOutput> : ModelOptions
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
    /// Gets or sets multiple teacher models (for ensemble distillation).
    /// </summary>
    /// <remarks>
    /// <para><b>For Ensemble Distillation:</b> Provide multiple teacher models.
    /// They will be automatically combined into an ensemble.</para>
    /// </remarks>
    public Vector<ITeacherModel<TInput, TOutput>>? Teachers { get; set; }

    /// <summary>
    /// Gets or sets the teacher IFullModel (recommended approach).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners (Recommended):</b> Pass your trained IFullModel directly.
    /// This is the standard way to provide a teacher model in the AiDotNet architecture.</para>
    /// <para>Example:
    /// <code>
    /// // After training
    /// var trainedModel = await builder.ConfigureModel(model).BuildAsync();
    ///
    /// // Use as teacher
    /// TeacherModel = trainedModel.Model
    /// </code>
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? TeacherModel { get; set; }

    /// <summary>
    /// Gets or sets the teacher model forward function (alternative approach).
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> If you have a trained model with a forward function,
    /// provide it and it will be automatically wrapped as a teacher.</para>
    /// <para>Example:
    /// <code>
    /// TeacherForward = input => myTrainedModel.Predict(input)
    /// </code>
    /// </para>
    /// </remarks>
    public Func<TInput, TOutput>? TeacherForward { get; set; }

    /// <summary>
    /// Gets or sets ensemble weights (if using multiple teachers).
    /// </summary>
    /// <remarks>
    /// <para>Optional weights for each teacher. Must sum to 1.0.
    /// If null, uniform weights are used.</para>
    /// </remarks>
    public Vector<double>? EnsembleWeights { get; set; }

    /// <summary>
    /// Gets or sets the distillation strategy instance (if using custom strategy).
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Provide a custom distillation strategy.
    /// If null, one will be created based on StrategyType.</para>
    /// </remarks>
    public IDistillationStrategy<T>? Strategy { get; set; }

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
    public Vector<string>? FeatureLayerPairs { get; set; }

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
    public Vector<string>? AttentionLayers { get; set; }

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
    /// Gets or sets whether to enable early stopping based on validation loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Production:</b> Stops training when validation loss stops improving.
    /// Prevents overfitting and saves compute time.</para>
    /// </remarks>
    public bool UseEarlyStopping { get; set; } = true;

    /// <summary>
    /// Gets or sets patience for early stopping (epochs without improvement).
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 3-10. Higher patience allows more time for improvement.</para>
    /// </remarks>
    public int EarlyStoppingPatience { get; set; } = 5;

    /// <summary>
    /// Gets or sets minimum improvement delta for early stopping.
    /// </summary>
    /// <remarks>
    /// <para>Loss must improve by at least this amount to count as improvement.
    /// Typical values: 0.001-0.01</para>
    /// </remarks>
    public double EarlyStoppingMinDelta { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets whether to save checkpoints during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Production:</b> Saves best model automatically.
    /// Essential for long-running training and recovery from failures.</para>
    /// </remarks>
    public bool SaveCheckpoints { get; set; } = false;

    /// <summary>
    /// Gets or sets checkpoint directory path (if checkpoints are enabled).
    /// </summary>
    /// <remarks>
    /// <para>If null, uses "./checkpoints" by default.</para>
    /// </remarks>
    public string? CheckpointDirectory { get; set; }

    /// <summary>
    /// Gets or sets checkpoint frequency (save every N epochs).
    /// </summary>
    /// <remarks>
    /// <para>Set to 1 to save after every epoch. Higher values save less frequently.</para>
    /// </remarks>
    public int CheckpointFrequency { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to only save the best model checkpoint.
    /// </summary>
    /// <remarks>
    /// <para>If true, only keeps the checkpoint with best validation loss.
    /// If false, keeps all checkpoints.</para>
    /// </remarks>
    public bool SaveOnlyBestCheckpoint { get; set; } = true;

    /// <summary>
    /// Gets or sets output dimension for models (if not inferrable from teacher).
    /// </summary>
    /// <remarks>
    /// <para>Usually inferred automatically. Set manually if needed.</para>
    /// </remarks>
    public int? OutputDimension { get; set; }

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
        if (LabelSmoothingFactor < 0 || LabelSmoothingFactor > 1)
            throw new ArgumentException("LabelSmoothingFactor must be between 0 and 1", nameof(LabelSmoothingFactor));
        if (SelfDistillationGenerations < 1)
            throw new ArgumentException("SelfDistillationGenerations must be at least 1", nameof(SelfDistillationGenerations));

        // Validate validation data consistency
        if (ValidationInputs != null || ValidationLabels != null)
        {
            // If either is provided, both must be provided
            if (ValidationInputs == null)
                throw new ArgumentNullException(nameof(ValidationInputs), "ValidationLabels is provided but ValidationInputs is null");
            if (ValidationLabels == null)
                throw new ArgumentNullException(nameof(ValidationLabels), "ValidationInputs is provided but ValidationLabels is null");

            // Both must be non-empty
            if (ValidationInputs.Length == 0)
                throw new ArgumentException("ValidationInputs cannot be empty", nameof(ValidationInputs));
            if (ValidationLabels.Length == 0)
                throw new ArgumentException("ValidationLabels cannot be empty", nameof(ValidationLabels));

            // Both must have the same length
            if (ValidationInputs.Length != ValidationLabels.Length)
                throw new ArgumentException(
                    $"ValidationInputs and ValidationLabels must have the same length. " +
                    $"ValidationInputs.Length = {ValidationInputs.Length}, ValidationLabels.Length = {ValidationLabels.Length}");
        }

        // If validation is enabled, validation data must be provided
        if (ValidateAfterEpoch && ValidationInputs == null)
            throw new ArgumentException("Validation data must be provided when ValidateAfterEpoch is true");
    }
}
