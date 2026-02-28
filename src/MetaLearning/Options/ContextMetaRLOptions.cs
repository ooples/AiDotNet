using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Context Meta-RL: context-conditioned meta-reinforcement
/// learning with attention-based aggregation.
/// </summary>
/// <remarks>
/// <para>
/// Context Meta-RL aggregates task context from support interactions using an
/// attention mechanism and uses the resulting context vector to modulate the
/// policy network parameters. Unlike PEARL's Gaussian posterior, this approach
/// uses deterministic attention-based aggregation with a learned query vector.
/// </para>
/// </remarks>
public class ContextMetaRLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Dimensionality of the context embedding. Default: 32.</summary>
    public int ContextDim { get; set; } = 32;

    /// <summary>Number of attention heads for context aggregation. Default: 4.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Modulation strength for context-conditioned parameter adjustment. Default: 0.1.</summary>
    public double ModulationStrength { get; set; } = 0.1;

    /// <summary>Attention temperature for softmax. Default: 1.0.</summary>
    public double AttentionTemperature { get; set; } = 1.0;

    public ContextMetaRLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && ContextDim > 0 && NumAttentionHeads > 0;
    public IMetaLearnerOptions<T> Clone() => new ContextMetaRLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        ContextDim = ContextDim, NumAttentionHeads = NumAttentionHeads,
        ModulationStrength = ModulationStrength, AttentionTemperature = AttentionTemperature
    };
}
