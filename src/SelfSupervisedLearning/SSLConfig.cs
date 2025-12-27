using AiDotNet.Enums;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Unified configuration for self-supervised learning with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Self-supervised learning (SSL) learns useful representations
/// from unlabeled data. This configuration controls how SSL pretraining works, including
/// the method to use, batch size, training epochs, and method-specific settings.</para>
///
/// <para><b>Key features:</b></para>
/// <list type="bullet">
/// <item>Automatic method selection (defaults to SimCLR)</item>
/// <item>Industry-standard defaults that work well out-of-the-box</item>
/// <item>Method-specific settings for fine-tuning (MoCo, BYOL, DINO, MAE)</item>
/// <item>Evaluation settings for linear probe and k-NN evaluation</item>
/// </list>
///
/// <para><b>Example - Simple usage with defaults:</b></para>
/// <code>
/// var result = builder
///     .ConfigureModel(encoder)
///     .ConfigureSelfSupervisedLearning()  // Uses SimCLR with defaults
///     .Build(unlabeledData);
/// </code>
///
/// <para><b>Example - Custom configuration:</b></para>
/// <code>
/// builder.ConfigureSelfSupervisedLearning(config =>
/// {
///     config.Method = SSLMethodType.MoCo;
///     config.PretrainingEpochs = 200;
///     config.BatchSize = 256;
///     config.MoCo = new MoCoConfig { QueueSize = 65536 };
/// });
/// </code>
/// </remarks>
public class SSLConfig
{
    // === Core Settings ===

    /// <summary>
    /// Gets or sets the SSL method to use.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>SSLMethodType.SimCLR</c></para>
    /// <para>If null, SimCLR is used as the default method.</para>
    /// </remarks>
    public SSLMethodType? Method { get; set; }

    /// <summary>
    /// Gets or sets the number of pretraining epochs.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>100</c></para>
    /// <para>SSL typically requires more epochs than supervised learning. Research papers often use 200-1000 epochs.</para>
    /// </remarks>
    public int? PretrainingEpochs { get; set; }

    /// <summary>
    /// Gets or sets the batch size for pretraining.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>256</c></para>
    /// <para>Larger batch sizes generally help contrastive methods (SimCLR benefits from 4096-8192).</para>
    /// </remarks>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the base learning rate.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.3</c> (with linear scaling rule)</para>
    /// <para>The effective learning rate is: base_lr * batch_size / 256</para>
    /// </remarks>
    public double? LearningRate { get; set; }

    /// <summary>
    /// Gets or sets whether to use cosine learning rate decay.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Cosine decay is standard for SSL pretraining.</para>
    /// </remarks>
    public bool? UseCosineDecay { get; set; }

    /// <summary>
    /// Gets or sets the number of warmup epochs.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>10</c></para>
    /// <para>Linear warmup helps stabilize early training.</para>
    /// </remarks>
    public int? WarmupEpochs { get; set; }

    /// <summary>
    /// Gets or sets the weight decay (L2 regularization).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>1e-4</c></para>
    /// </remarks>
    public double? WeightDecay { get; set; }

    // === Projection Head Settings ===

    /// <summary>
    /// Gets or sets the output dimension of the projection head.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>128</c></para>
    /// <para>Common values: 128 (SimCLR, MoCo), 256 (BYOL), 2048 (Barlow Twins).</para>
    /// </remarks>
    public int? ProjectorOutputDimension { get; set; }

    /// <summary>
    /// Gets or sets the hidden dimension of the projection head MLP.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>2048</c></para>
    /// </remarks>
    public int? ProjectorHiddenDimension { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the projection head.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>2</c></para>
    /// <para>Common values: 2 (SimCLR, MoCo), 3 (BYOL with predictor).</para>
    /// </remarks>
    public int? ProjectorLayers { get; set; }

    // === Temperature Settings ===

    /// <summary>
    /// Gets or sets the temperature parameter for contrastive loss.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.07</c></para>
    /// <para>Lower temperature makes the distribution sharper (harder negatives).</para>
    /// <para>Common values: 0.07 (MoCo), 0.1 (SimCLR), 0.5 (some variants).</para>
    /// </remarks>
    public double? Temperature { get; set; }

    /// <summary>
    /// Gets or sets whether to use temperature scheduling.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Some methods benefit from scheduling temperature during training.</para>
    /// </remarks>
    public bool? UseTemperatureScheduling { get; set; }

    // === Method-Specific Settings ===

    /// <summary>
    /// Gets or sets MoCo-specific configuration.
    /// </summary>
    public MoCoConfig? MoCo { get; set; }

    /// <summary>
    /// Gets or sets BYOL-specific configuration.
    /// </summary>
    public BYOLConfig? BYOL { get; set; }

    /// <summary>
    /// Gets or sets DINO-specific configuration.
    /// </summary>
    public DINOConfig? DINO { get; set; }

    /// <summary>
    /// Gets or sets MAE-specific configuration.
    /// </summary>
    public MAEConfig? MAE { get; set; }

    /// <summary>
    /// Gets or sets Barlow Twins-specific configuration.
    /// </summary>
    public BarlowTwinsConfig? BarlowTwins { get; set; }

    // === Evaluation Settings ===

    /// <summary>
    /// Gets or sets whether to run linear evaluation during/after pretraining.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Linear evaluation trains a linear classifier on frozen features to measure representation quality.</para>
    /// </remarks>
    public bool? EnableLinearEvaluation { get; set; }

    /// <summary>
    /// Gets or sets the frequency of linear evaluation (epochs).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>10</c></para>
    /// <para>Set to 0 to only evaluate at the end of training.</para>
    /// </remarks>
    public int? LinearEvaluationFrequency { get; set; }

    /// <summary>
    /// Gets or sets whether to run k-NN evaluation.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>k-NN evaluation is faster than linear and doesn't require training.</para>
    /// </remarks>
    public bool? EnableKNNEvaluation { get; set; }

    /// <summary>
    /// Gets or sets the number of neighbors for k-NN evaluation.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>20</c></para>
    /// </remarks>
    public int? KNNNeighbors { get; set; }

    // === Checkpointing Settings ===

    /// <summary>
    /// Gets or sets the checkpoint save frequency (epochs).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>20</c></para>
    /// <para>Set to 0 to disable automatic checkpointing.</para>
    /// </remarks>
    public int? CheckpointFrequency { get; set; }

    /// <summary>
    /// Gets or sets the path to save checkpoints.
    /// </summary>
    public string? CheckpointPath { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Creates a new SSL configuration with industry-standard defaults.
    /// </summary>
    public SSLConfig()
    {
    }

    /// <summary>
    /// Gets the configuration as a dictionary for logging or serialization.
    /// </summary>
    /// <returns>A dictionary containing all configuration values.</returns>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>();

        if (Method.HasValue) config["method"] = Method.Value.ToString();
        if (PretrainingEpochs.HasValue) config["pretrainingEpochs"] = PretrainingEpochs.Value;
        if (BatchSize.HasValue) config["batchSize"] = BatchSize.Value;
        if (LearningRate.HasValue) config["learningRate"] = LearningRate.Value;
        if (Temperature.HasValue) config["temperature"] = Temperature.Value;
        if (ProjectorOutputDimension.HasValue) config["projectorOutputDimension"] = ProjectorOutputDimension.Value;
        if (EnableLinearEvaluation.HasValue) config["enableLinearEvaluation"] = EnableLinearEvaluation.Value;
        if (EnableKNNEvaluation.HasValue) config["enableKNNEvaluation"] = EnableKNNEvaluation.Value;
        if (Seed.HasValue) config["seed"] = Seed.Value;

        if (MoCo is not null) config["moco"] = MoCo.GetConfiguration();
        if (BYOL is not null) config["byol"] = BYOL.GetConfiguration();
        if (DINO is not null) config["dino"] = DINO.GetConfiguration();
        if (MAE is not null) config["mae"] = MAE.GetConfiguration();
        if (BarlowTwins is not null) config["barlowTwins"] = BarlowTwins.GetConfiguration();

        return config;
    }
}
