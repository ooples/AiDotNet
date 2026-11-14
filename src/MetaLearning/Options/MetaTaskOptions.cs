using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MetaTask algorithm, an implementation of MetaMix (Chen et al., ICPR 2020).
/// </summary>
/// <remarks>
/// MetaMix keeps the MAML inner loop on the support set and replaces each episode's query set by
/// virtual examples mixed from random pairs of examples with a Beta-distributed coefficient.
/// </remarks>
public class MetaTaskOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Optimizer for the meta-update. When set, it takes the outer step at its own learning rate (so
    /// <see cref="OuterLearningRate"/> no longer applies) and keeps its state across meta-iterations; null uses
    /// plain SGD at <see cref="OuterLearningRate"/>. It must not be the same instance as <see cref="InnerOptimizer"/>.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Optimizer for task adaptation. When set, it takes each inner step at its own learning rate (so
    /// <see cref="InnerLearningRate"/> no longer applies) and is reset at the start of every adaptation; null uses
    /// plain SGD at <see cref="InnerLearningRate"/>.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Alpha parameter of the symmetric Beta(α, α) distribution that draws the mixup coefficient λ
    /// (MetaMix Eq. 4-5). The paper's experiments use α = 1, i.e. λ uniform on [0, 1]; values of 1 and
    /// above performed alike, values below 1 slightly worse. Default: 1.0.
    /// </summary>
    public double InterpolationAlpha { get; set; } = 1.0;

    /// <summary>
    /// Number Z of virtual (mixed) examples generated per episode (MetaMix Algorithm 1, line 14).
    /// 0 means one per query example, which keeps the outer batch size of MAML. Default: 0.
    /// </summary>
    public int NumMixedPairs { get; set; } = 0;

    /// <summary>
    /// Which set the two mixed examples are drawn from. The paper's MetaMix mixes pairs from the query
    /// set only (its ablation, Table III, found that best). Default: <see cref="MetaMixSource.Query"/>.
    /// </summary>
    public MetaMixSource MixSource { get; set; } = MetaMixSource.Query;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public MetaTaskOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 &&
        InterpolationAlpha > 0 && NumMixedPairs >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new MetaTaskOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        InterpolationAlpha = InterpolationAlpha, NumMixedPairs = NumMixedPairs, MixSource = MixSource
    };
}

/// <summary>
/// Selects the sets MetaMix draws its example pairs from.
/// </summary>
public enum MetaMixSource
{
    /// <summary>Both examples of a pair come from the query set (the paper's MetaMix).</summary>
    Query,

    /// <summary>The first example comes from the support set and the second from the query set.</summary>
    SupportAndQuery,
}
