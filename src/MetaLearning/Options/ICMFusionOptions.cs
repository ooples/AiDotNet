using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for ICM-Fusion (In-Context Meta-Optimized LoRA Fusion, 2025).
/// </summary>
/// <remarks>
/// <para>
/// ICM-Fusion fuses multiple task-specific parameter deltas by encoding them into a latent
/// space via a Fusion-VAE, then reconstructing the fused adapter. The VAE is meta-learned
/// so that task vector arithmetic in latent space resolves inter-weight conflicts.
/// </para>
/// </remarks>
public class ICMFusionOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Dimensionality of the VAE latent space. Default: 32.
    /// </summary>
    public int LatentDim { get; set; } = 32;

    /// <summary>
    /// Number of task-specific components to maintain for fusion. Default: 3.
    /// </summary>
    public int NumFusionComponents { get; set; } = 3;

    /// <summary>
    /// KL divergence weight in the VAE loss: L = L_recon + KLWeight * L_KL. Default: 0.005.
    /// </summary>
    public double KLWeight { get; set; } = 0.005;

    /// <summary>
    /// Exponential decay for older fusion components. Default: 0.9.
    /// </summary>
    public double FusionDecay { get; set; } = 0.9;

    public ICMFusionOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && LatentDim > 0;
    public IMetaLearnerOptions<T> Clone() => new ICMFusionOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        LatentDim = LatentDim, NumFusionComponents = NumFusionComponents, KLWeight = KLWeight, FusionDecay = FusionDecay
    };
}
