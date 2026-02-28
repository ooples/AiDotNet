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

    public HyperNeRFMetaOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
                              && NumFrequencyBands > 0 && LatentDim > 0;
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
