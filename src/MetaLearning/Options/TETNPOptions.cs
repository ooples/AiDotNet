using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Translation-Equivariant Transformer Neural Process (TE-TNP).
/// Extends TNP with relative positional encodings for translation equivariance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> TE-TNP predicts functions like TNP but ensures that shifting
/// all inputs by the same amount shifts all outputs correspondingly. This is achieved by
/// using relative distances between points rather than absolute positions in the attention mechanism.</para>
/// </remarks>
public class TETNPOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Gets or sets the representation dimensionality.</summary>
    public int RepresentationDim { get; set; } = 128;

    /// <summary>
    /// Number of frequency bands for sinusoidal relative positional encoding.
    /// Higher values capture finer positional distinctions.
    /// </summary>
    public int NumFrequencyBands { get; set; } = 8;

    /// <summary>
    /// Number of attention heads for the equivariant self-attention.
    /// </summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Weight for the equivariance regularization loss that encourages
    /// translation-equivariant behavior during training.
    /// </summary>
    public double EquivarianceRegWeight { get; set; } = 0.01;

    public TETNPOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new TETNPOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        RepresentationDim = RepresentationDim, NumFrequencyBands = NumFrequencyBands,
        NumHeads = NumHeads, EquivarianceRegWeight = EquivarianceRegWeight
    };
}
