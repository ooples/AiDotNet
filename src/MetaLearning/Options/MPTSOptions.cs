using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MPTS (Meta-learning with Progressive Task-Specific adaptation) algorithm.
/// </summary>
/// <remarks>
/// MPTS groups model parameters into blocks and learns per-block adaptation priorities.
/// High-priority groups are adapted in early inner loop steps, while lower-priority groups
/// are gradually unfrozen as adaptation progresses — a progressive unfreezing strategy
/// that reduces overfitting on small support sets.
/// </remarks>
public class MPTSOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of parameter groups for progressive adaptation. Parameters are evenly
    /// divided into groups, with earlier groups having higher priority. Default: 4.
    /// </summary>
    public int NumParamGroups { get; set; } = 4;

    /// <summary>
    /// Rate at which group activation decays from high to low priority.
    /// Group g is active at step k if: sigmoid((k/K - g/G) * 10 / PriorityDecayRate) > 0.5.
    /// Lower values mean more progressive (gradual) unfreezing. Default: 0.5.
    /// </summary>
    public double PriorityDecayRate { get; set; } = 0.5;

    /// <summary>
    /// L2 regularization weight between groups to encourage coherent adaptation.
    /// Default: 0.01.
    /// </summary>
    public double GroupRegWeight { get; set; } = 0.01;

    public MPTSOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new MPTSOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumParamGroups = NumParamGroups, PriorityDecayRate = PriorityDecayRate,
        GroupRegWeight = GroupRegWeight
    };
}
