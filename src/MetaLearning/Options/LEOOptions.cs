using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Latent Embedding Optimization (LEO) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// LEO performs meta-learning by learning a low-dimensional latent space for model parameters.
/// Instead of adapting the full model parameters directly (like MAML), LEO:
/// </para>
/// <list type="number">
/// <item>Encodes support examples into a latent representation</item>
/// <item>Decodes the latent code into model parameters</item>
/// <item>Performs gradient descent in the latent space during adaptation</item>
/// <item>Decodes the adapted latent code to get final model parameters</item>
/// </list>
/// <para><b>For Beginners:</b> Imagine you have a very large model with millions of parameters.
/// Updating all of them during few-shot learning is slow and can lead to overfitting.
/// LEO learns to "compress" these parameters into a much smaller space (like 64 numbers instead
/// of millions). Adaptation happens in this compressed space, which is faster and more
/// robust to overfitting.
/// </para>
/// <para>
/// <b>Key Insight:</b> Not all parameter configurations make sense for neural networks.
/// By learning a latent space, LEO restricts adaptation to the "manifold" of sensible
/// parameter settings, preventing bad updates.
/// </para>
/// </remarks>
public class LEOOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (feature encoder) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a feature encoder. LEO will learn to generate
    /// task-specific classifier parameters from support examples using this encoder.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses cross-entropy loss internally).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter (outer loop) updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates.
    /// Default: null (uses SGD with InnerLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (latent space adaptation).
    /// </summary>
    /// <value>Default is 1.0 (full step in latent space).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Since adaptation happens in a learned latent space,
    /// the learning rate can often be larger than in parameter space. The latent
    /// space is designed to make adaptation easier.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-update).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps in latent space.
    /// </summary>
    /// <value>Default is 5.</value>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default is 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default is 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate the meta-learner.
    /// </summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save model checkpoints.
    /// </summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is false (LEO uses second-order gradients through the decoder).</value>
    public bool UseFirstOrder { get; set; } = false;

    #endregion

    #region LEO-Specific Properties

    /// <summary>
    /// Gets or sets the dimensionality of the latent space.
    /// </summary>
    /// <value>Default is 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how "compressed" the parameter space is.
    /// Smaller values are more efficient but may limit what the model can represent.
    /// Typical values are 32-128.
    /// </para>
    /// </remarks>
    public int LatentDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the dimension of the feature embedding.
    /// </summary>
    /// <value>Default is 512.</value>
    public int EmbeddingDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the hidden dimension for encoder/decoder networks.
    /// </summary>
    /// <value>Default is 256.</value>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the KL divergence weight for the latent space regularization.
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> LEO uses a variational approach where the latent
    /// space is regularized to be close to a prior distribution (usually Gaussian).
    /// This weight controls how strongly we enforce this regularization.
    /// </para>
    /// </remarks>
    public double KLWeight { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the entropy weight for regularizing the decoder.
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double EntropyWeight { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets whether to use orthogonal initialization for the decoder.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseOrthogonalInit { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to share the encoder across all classes.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, the same encoder is used to generate
    /// latent codes for all classes. If false, each class has its own encoder,
    /// which allows more flexibility but requires more parameters.
    /// </para>
    /// </remarks>
    public bool ShareEncoder { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use a relation network for encoding.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A relation network considers relationships between
    /// all support examples when generating the latent code, which can help when
    /// examples within a class are diverse.
    /// </para>
    /// </remarks>
    public bool UseRelationEncoder { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for the encoder and decoder.
    /// </summary>
    /// <value>Default is 0.0 (no dropout).</value>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the L2 regularization strength.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double L2Regularization { get; set; } = 0.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the LEOOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The feature encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public LEOOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all LEO configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               LatentDimension > 0 &&
               EmbeddingDimension > 0 &&
               NumClasses > 0 &&
               HiddenDimension > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               KLWeight >= 0 &&
               EntropyWeight >= 0;
    }

    /// <summary>
    /// Creates a deep copy of the LEO options.
    /// </summary>
    /// <returns>A new LEOOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new LEOOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer,
            DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate,
            OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps,
            MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            UseFirstOrder = UseFirstOrder,
            LatentDimension = LatentDimension,
            EmbeddingDimension = EmbeddingDimension,
            NumClasses = NumClasses,
            HiddenDimension = HiddenDimension,
            KLWeight = KLWeight,
            EntropyWeight = EntropyWeight,
            UseOrthogonalInit = UseOrthogonalInit,
            ShareEncoder = ShareEncoder,
            UseRelationEncoder = UseRelationEncoder,
            DropoutRate = DropoutRate,
            L2Regularization = L2Regularization
        };
    }

    #endregion
}
