using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for DiscoRL: Discovery-based meta-RL with skill discovery.
/// </summary>
/// <remarks>
/// <para>
/// DiscoRL discovers reusable "skills" (parameter subspaces) during meta-training
/// and combines them for new task adaptation. Each skill corresponds to a low-rank
/// direction in parameter space. A skill selector (gating network) chooses which
/// skills to activate for each task based on early gradient signals.
/// </para>
/// </remarks>
public class DiscoRLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Number of discoverable skills. Default: 8.</summary>
    public int NumSkills { get; set; } = 8;

    /// <summary>Rank of each skill direction in parameter space. Default: 4.</summary>
    public int SkillRank { get; set; } = 4;

    /// <summary>Temperature for skill selection softmax. Default: 1.0.</summary>
    public double SelectionTemperature { get; set; } = 1.0;

    /// <summary>Entropy bonus to encourage diverse skill usage. Default: 0.01.</summary>
    public double SkillEntropyBonus { get; set; } = 0.01;

    public DiscoRLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumSkills > 0 && SkillRank > 0;
    public IMetaLearnerOptions<T> Clone() => new DiscoRLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumSkills = NumSkills, SkillRank = SkillRank, SelectionTemperature = SelectionTemperature,
        SkillEntropyBonus = SkillEntropyBonus
    };
}
