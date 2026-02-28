using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-DM: Applications of Diffusion Models on Few-Shot Learning
/// (Hu et al., ICIP 2024).
/// </summary>
/// <remarks>
/// <para>
/// Meta-DM uses a DDPM-style diffusion model as a data augmentation module for few-shot learning.
/// It generates synthetic support samples conditioned on the existing few-shot support set, then
/// trains on the enriched dataset. This is a modular augmentation strategy composable with any
/// gradient-based meta-learning algorithm.
/// </para>
/// </remarks>
public class MetaDMOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of synthetic samples to generate per class for augmentation. Default: 5.
    /// </summary>
    public int SyntheticSamplesPerClass { get; set; } = 5;

    /// <summary>
    /// Number of diffusion timesteps for generation. Default: 50.
    /// </summary>
    public int DiffusionTimesteps { get; set; } = 50;

    /// <summary>
    /// Number of denoising steps for generation (≤ DiffusionTimesteps). Default: 10.
    /// </summary>
    public int DenoisingSteps { get; set; } = 10;

    /// <summary>
    /// Starting noise schedule parameter. Default: 0.0001.
    /// </summary>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Ending noise schedule parameter. Default: 0.02.
    /// </summary>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Dimensionality of the prototype embeddings for distribution matching. Default: 32.
    /// </summary>
    public int PrototypeDim { get; set; } = 32;

    /// <summary>
    /// Weight of the distribution matching loss relative to the task loss. Default: 0.1.
    /// </summary>
    public double MatchingWeight { get; set; } = 0.1;

    public MetaDMOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && DiffusionTimesteps > 0;
    public IMetaLearnerOptions<T> Clone() => new MetaDMOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        SyntheticSamplesPerClass = SyntheticSamplesPerClass, DiffusionTimesteps = DiffusionTimesteps,
        DenoisingSteps = DenoisingSteps, BetaStart = BetaStart, BetaEnd = BetaEnd,
        PrototypeDim = PrototypeDim, MatchingWeight = MatchingWeight
    };
}
