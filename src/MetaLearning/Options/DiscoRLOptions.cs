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
    /// <summary>Whether to use the first-order approximation of the meta-gradient. Default: true.</summary>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function for meta-training; null uses the default multi-class classification loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: its meta-update is built in (scaled by <see cref="OuterLearningRate"/>). A
    /// supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: any per-task adaptation is built in (scaled by <see cref="InnerLearningRate"/>).
    /// A supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>Number of discoverable skills. Default: 8.</summary>
    public int NumSkills { get; set; } = 8;

    /// <summary>Rank of each skill direction in parameter space. Default: 4.</summary>
    public int SkillRank { get; set; } = 4;

    /// <summary>Temperature for skill selection softmax. Default: 1.0.</summary>
    public double SelectionTemperature { get; set; } = 1.0;

    /// <summary>Entropy bonus to encourage diverse skill usage. Default: 0.01.</summary>
    public double SkillEntropyBonus { get; set; } = 0.01;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public DiscoRLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 &&
        NumSkills > 0 && SkillRank > 0 && SelectionTemperature > 0 && SkillEntropyBonus >= 0;
    /// <inheritdoc/>
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
