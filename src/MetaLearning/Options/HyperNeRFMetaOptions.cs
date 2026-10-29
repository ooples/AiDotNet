using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>Configuration options for HyperNeRF Meta-learning.</summary>
/// <remarks>
/// <para>
/// Combines hypernetwork conditioning with NeRF-style positional encoding. Each parameter
/// index is positionally encoded using sinusoidal frequencies, and combined with a task
/// latent code to produce per-parameter learning rate modulation. This gives the
/// hypernetwork structural awareness of parameter positions within the network.
/// </para>
/// </remarks>
public class HyperNeRFMetaOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of sinusoidal frequency bands for positional encoding of
    /// parameter indices. Encoding dimension = 2 * NumFrequencyBands + 1.
    /// Default: 4.
    /// </summary>
    public int NumFrequencyBands { get; set; } = 4;

    /// <summary>
    /// Dimension of the task-specific latent code derived from gradient statistics.
    /// Default: 16.
    /// </summary>
    public int LatentDim { get; set; } = 16;

    /// <summary>
    /// L2 regularization weight on the conditioning MLP weights to
    /// prevent overfitting the hypernetwork. Default: 0.01.
    /// </summary>
    public double ConditioningRegWeight { get; set; } = 0.01;

    /// <summary>
    /// Strength of frequency-based conditioning modulation. The final
    /// modulation factor is: 1 + ConditioningStrength * (mlpOutput - 0.5).
    /// Default: 1.0.
    /// </summary>
    public double ConditioningStrength { get; set; } = 1.0;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public HyperNeRFMetaOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        NumFrequencyBands > 0 &&
        LatentDim > 0 &&
        ConditioningRegWeight >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new HyperNeRFMetaOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumFrequencyBands = NumFrequencyBands, LatentDim = LatentDim,
        ConditioningRegWeight = ConditioningRegWeight, ConditioningStrength = ConditioningStrength
    };
}
