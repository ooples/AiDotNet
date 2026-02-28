using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for JMP (Joint Multi-Phase meta-learning).
/// </summary>
/// <remarks>
/// JMP uses a multi-phase inner loop with separate learning rates and regularization.
/// Phase 1 (coarse) uses a higher learning rate for fast, rough adaptation.
/// Phase 2 (fine) uses a lower learning rate with stronger regularization toward the
/// Phase 1 result for careful refinement.
/// </remarks>
public class JMPOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Fraction of adaptation steps in Phase 1 (coarse). Default: 0.4.</summary>
    public double Phase1Fraction { get; set; } = 0.4;

    /// <summary>Learning rate multiplier for Phase 1 (coarse). Default: 2.0.</summary>
    public double Phase1LRMultiplier { get; set; } = 2.0;

    /// <summary>Learning rate multiplier for Phase 2 (fine). Default: 0.5.</summary>
    public double Phase2LRMultiplier { get; set; } = 0.5;

    /// <summary>L2 regularization during Phase 2 toward Phase 1 result. Default: 0.1.</summary>
    public double PhaseRegWeight { get; set; } = 0.1;

    public JMPOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new JMPOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        Phase1Fraction = Phase1Fraction, Phase1LRMultiplier = Phase1LRMultiplier,
        Phase2LRMultiplier = Phase2LRMultiplier, PhaseRegWeight = PhaseRegWeight
    };
}
