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
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    public double InnerLearningRate { get; set; } = 0.01;
    public double OuterLearningRate { get; set; } = 0.001;
    public int AdaptationSteps { get; set; } = 5;
    public int MetaBatchSize { get; set; } = 4;
    public int NumMetaIterations { get; set; } = 1000;
    public double? GradientClipThreshold { get; set; } = 10.0;
    public int? RandomSeed { get => Seed; set => Seed = value; }
    public int EvaluationTasks { get; set; } = 100;
    public int EvaluationFrequency { get; set; } = 100;
    public bool EnableCheckpointing { get; set; } = false;
    public int CheckpointFrequency { get; set; } = 500;
    public bool UseFirstOrder { get; set; } = true;
    public ILossFunction<T>? LossFunction { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
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

    public ACLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
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
