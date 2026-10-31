using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MetaTask (Meta-learned Task Augmentation) algorithm.
/// </summary>
/// <remarks>
/// MetaTask generates synthetic tasks by interpolating gradients between pairs of real tasks
/// using Beta-distributed mixing coefficients. The synthetic tasks augment the task distribution,
/// improving generalization of the meta-learned initialization.
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
    /// Alpha parameter for Beta(α,α) distribution used to sample interpolation coefficients.
    /// Higher values concentrate mixing around λ=0.5; lower values allow extreme mixing.
    /// Default: 2.0.
    /// </summary>
    public double InterpolationAlpha { get; set; } = 2.0;

    /// <summary>
    /// Number of synthetic (interpolated) tasks generated per meta-batch.
    /// These are added to the real tasks in the meta-objective. Default: 2.
    /// </summary>
    public int NumSyntheticTasks { get; set; } = 2;

    /// <summary>
    /// Weight on the synthetic task losses relative to real task losses.
    /// L_meta = L_real + SyntheticWeight * L_synthetic. Default: 0.5.
    /// </summary>
    public double SyntheticWeight { get; set; } = 0.5;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public MetaTaskOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 &&
        InterpolationAlpha > 0 && NumSyntheticTasks > 0 && SyntheticWeight >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new MetaTaskOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        InterpolationAlpha = InterpolationAlpha, NumSyntheticTasks = NumSyntheticTasks,
        SyntheticWeight = SyntheticWeight
    };
}
