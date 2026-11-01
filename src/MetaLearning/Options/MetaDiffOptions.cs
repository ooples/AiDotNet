using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot
/// Learning (Zhang et al., AAAI 2024).
/// </summary>
/// <remarks>
/// <para>
/// MetaDiff reframes the inner-loop gradient descent of meta-learning as a reverse diffusion
/// process over model weights. A task-conditional denoising network iteratively removes noise
/// to produce task-specific weights, conditioned on support-set features.
/// </para>
/// </remarks>
public class MetaDiffOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>The model whose parameters the meta-learner trains to adapt quickly to new tasks.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    /// <inheritdoc/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <summary>Learning rate of the outer (meta) update. Default: 0.0001.</summary>
    public double OuterLearningRate { get; set; } = 0.0001;
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
    /// Number of diffusion timesteps for the full forward/reverse process. Default: 100.
    /// During inference, fewer steps can be used (controlled by <see cref="SamplingSteps"/>).
    /// </summary>
    public int DiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Number of denoising steps used during inference (≤ DiffusionSteps). Default: 20.
    /// </summary>
    public int SamplingSteps { get; set; } = 20;

    /// <summary>
    /// Starting value of the linear noise schedule β_1. Default: 0.0001.
    /// </summary>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Ending value of the linear noise schedule β_T. Default: 0.02.
    /// </summary>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Dimensionality of the task conditioning vector computed from support features. Default: 64.
    /// </summary>
    public int TaskConditionDim { get; set; } = 64;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public MetaDiffOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && DiffusionSteps > 0 && SamplingSteps > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new MetaDiffOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        DiffusionSteps = DiffusionSteps, SamplingSteps = SamplingSteps, BetaStart = BetaStart,
        BetaEnd = BetaEnd, TaskConditionDim = TaskConditionDim
    };
}
