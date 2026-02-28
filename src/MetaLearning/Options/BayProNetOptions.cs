using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for BayProNet: Bayesian Prototypical Networks for few-shot learning
/// with uncertainty estimation.
/// </summary>
/// <remarks>
/// <para>
/// BayProNet extends Prototypical Networks by modeling class prototypes as Gaussian
/// distributions N(μ_c, σ²_c) rather than point estimates. The prototype mean and
/// variance are computed from support set embeddings, and classification uses the
/// expected negative log-likelihood under the prototype distribution rather than
/// simple Euclidean distance.
/// </para>
/// <para><b>Key equations:</b>
/// <code>
/// Prototype distribution: μ_c = mean(f(x_i)), log(σ²_c) = g(support_c)
/// Predictive distribution: p(y=c|x) ∝ exp(-d_Mahal(f(x), μ_c, σ²_c) - 0.5*Σ log(σ²_c))
/// d_Mahal = Σ_d (f(x)_d - μ_c,d)² / σ²_c,d
/// </code>
/// </para>
/// </remarks>
public class BayProNetOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Dimensionality of prototype embeddings. Default: 32.
    /// </summary>
    public int EmbeddingDim { get; set; } = 32;

    /// <summary>
    /// Initial log-variance for prototype distributions. Default: -2.0 (σ² ≈ 0.14).
    /// </summary>
    public double InitialPrototypeLogVar { get; set; } = -2.0;

    /// <summary>
    /// Weight for the KL divergence between prototype posteriors and a unit Gaussian prior. Default: 0.01.
    /// </summary>
    public double KLWeight { get; set; } = 0.01;

    /// <summary>
    /// Temperature scaling for the softmax over Mahalanobis distances. Default: 1.0.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    public BayProNetOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && EmbeddingDim > 0 && Temperature > 0;
    public IMetaLearnerOptions<T> Clone() => new BayProNetOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        EmbeddingDim = EmbeddingDim, InitialPrototypeLogVar = InitialPrototypeLogVar,
        KLWeight = KLWeight, Temperature = Temperature
    };
}
