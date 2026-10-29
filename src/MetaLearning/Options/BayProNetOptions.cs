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
    /// Not used by this algorithm yet: its meta-update is built in (scaled by <see cref="OuterLearningRate"/>). A
    /// supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: any per-task adaptation is built in (scaled by <see cref="InnerLearningRate"/>).
    /// A supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
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

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public BayProNetOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        EmbeddingDim > 0 &&
        Temperature > 0 &&
        KLWeight >= 0;
    /// <inheritdoc/>
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
