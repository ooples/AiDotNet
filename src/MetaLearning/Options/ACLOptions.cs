using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the ACL (Adaptive Continual Learning) algorithm.
/// </summary>
/// <remarks>
/// ACL learns task-specific parameter importance masks that protect critical weights
/// from catastrophic forgetting. The importance is estimated from gradient magnitudes
/// accumulated via exponential moving average across adaptation steps.
/// </remarks>
public class ACLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>The model whose parameters the meta-learner trains to adapt quickly to new tasks.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    /// <inheritdoc/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc/>
    public double OuterLearningRate { get; set; } = 0.001;
    /// <inheritdoc/>
    public int AdaptationSteps { get; set; } = 5;
    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <inheritdoc/>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <inheritdoc/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function used in the inner and outer loops; null uses the meta-model's default loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Optimizer for the outer (meta) update; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Optimizer for the inner, per-task adaptation loop; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// L1 penalty coefficient on the parameter importance masks to encourage sparsity.
    /// Higher values produce sparser masks (fewer protected parameters). Default: 0.01.
    /// </summary>
    public double MaskSparsityPenalty { get; set; } = 0.01;

    /// <summary>
    /// EMA decay rate for accumulating parameter importance across tasks.
    /// Values close to 1.0 give more weight to historical importance. Default: 0.9.
    /// </summary>
    public double ImportanceDecay { get; set; } = 0.9;

    /// <summary>
    /// How much to scale down the learning rate for parameters deemed important.
    /// Effective inner LR = InnerLearningRate / (1 + ProtectionStrength * importance).
    /// Default: 10.0.
    /// </summary>
    public double ProtectionStrength { get; set; } = 10.0;

    /// <summary>
    /// Regularization weight pulling adapted parameters toward the initial (pre-task) values
    /// for protected parameters. Default: 0.1.
    /// </summary>
    public double ElasticRegWeight { get; set; } = 0.1;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public ACLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 &&
        MaskSparsityPenalty >= 0 && ImportanceDecay >= 0 && ImportanceDecay <= 1 &&
        ProtectionStrength >= 0 && ElasticRegWeight >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new ACLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        MaskSparsityPenalty = MaskSparsityPenalty, ImportanceDecay = ImportanceDecay,
        ProtectionStrength = ProtectionStrength, ElasticRegWeight = ElasticRegWeight
    };
}
