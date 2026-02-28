using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for PEARL: Probabilistic Embeddings for Actor-critic RL
/// (Rakelly et al., ICML 2019).
/// </summary>
/// <remarks>
/// <para>
/// PEARL uses a probabilistic context encoder to infer a latent task variable z from
/// transition data. The context encoder produces a Gaussian posterior q(z|c) that
/// conditions the policy and value function. Task inference is amortized and
/// gradient-free at test time — the encoder simply processes new transitions.
/// </para>
/// </remarks>
public class PEARLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Dimensionality of the latent task variable z. Default: 16.</summary>
    public int LatentDim { get; set; } = 16;

    /// <summary>KL divergence weight for the posterior regularization. Default: 0.1.</summary>
    public double KLWeight { get; set; } = 0.1;

    /// <summary>Number of posterior samples for training. Default: 5.</summary>
    public int NumPosteriorSamples { get; set; } = 5;

    /// <summary>Context encoder hidden dimension. Default: 64.</summary>
    public int EncoderHiddenDim { get; set; } = 64;

    public PEARLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && LatentDim > 0 && NumPosteriorSamples > 0;
    public IMetaLearnerOptions<T> Clone() => new PEARLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        LatentDim = LatentDim, KLWeight = KLWeight, NumPosteriorSamples = NumPosteriorSamples,
        EncoderHiddenDim = EncoderHiddenDim
    };
}
