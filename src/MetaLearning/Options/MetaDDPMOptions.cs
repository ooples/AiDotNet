using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models.
/// </summary>
/// <remarks>
/// <para>
/// Meta-DDPM extends the DDPM framework to meta-learning by meta-learning a noise prediction
/// network that generates task-specific model weights conditioned on support set embeddings.
/// Unlike MetaDiff (which models weight deltas), Meta-DDPM directly generates the full
/// adapted weight vector using the DDPM generative framework with a learned linear noise
/// schedule and task-conditional denoising.
/// </para>
/// </remarks>
public class MetaDDPMOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>The model whose parameters the meta-learner trains to adapt quickly to new tasks.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    /// <inheritdoc/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc/>
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
    /// Total timesteps in the DDPM diffusion process. Default: 200.
    /// </summary>
    public int NumTimesteps { get; set; } = 200;

    /// <summary>
    /// Starting beta for the linear noise schedule. Default: 0.0001.
    /// </summary>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Ending beta for the linear noise schedule. Default: 0.02.
    /// </summary>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Number of denoising steps used during generation (can be less than NumTimesteps). Default: 20.
    /// </summary>
    public int SamplingSteps { get; set; } = 20;

    /// <summary>
    /// Dimensionality of the task conditioning vector. Default: 64.
    /// </summary>
    public int TaskConditionDim { get; set; } = 64;

    /// <summary>
    /// Weight for EMA (exponential moving average) of model parameters for stable generation.
    /// Default: 0.999.
    /// </summary>
    public double EmaDecay { get; set; } = 0.999;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public MetaDDPMOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        NumTimesteps > 0 &&
        SamplingSteps > 0 && SamplingSteps <= NumTimesteps &&
        TaskConditionDim > 0 &&
        EmaDecay > 0 && EmaDecay < 1;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new MetaDDPMOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumTimesteps = NumTimesteps, BetaStart = BetaStart, BetaEnd = BetaEnd,
        SamplingSteps = SamplingSteps, TaskConditionDim = TaskConditionDim, EmaDecay = EmaDecay
    };
}
