using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

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
    /// Gets or sets the initial latent step size; LEO then learns one step size per latent dimension.
    /// </summary>
    /// <value>Default is 1.0, the paper's initialisation (Appendix B.4).</value>
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
    /// <value>Default is 0.00043653954, the paper's miniImageNet 1-shot value.</value>
    /// <remarks>
    /// <para>Table 6's value, found by random search on the miniImageNet 1-shot meta-validation set; its many
    /// significant figures are the search's, not a derivation. Other benchmarks chose other values.</para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.00043653954;

    /// <summary>
    /// Gets or sets the number of adaptation steps in latent space.
    /// </summary>
    /// <value>Default is 5, the paper's setting (Appendix B.4).</value>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of fine-tuning steps taken directly on the generated classifier weights after the
    /// latent steps.
    /// </summary>
    /// <value>Default is 5, the paper's setting (section 4.2.3, Appendix B.4). Zero skips fine-tuning.</value>
    public int FineTuningSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the initial fine-tuning step size; LEO then learns one step size per embedding dimension.
    /// </summary>
    /// <value>Default is 0.001, the paper's initialisation (Appendix B.4).</value>
    public double FineTuningLearningRate { get; set; } = 0.001;

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
    /// Gets or sets the maximum norm of each meta-gradient block.
    /// </summary>
    /// <value>Default is 0.1, the paper's clipping threshold (Appendix B.4).</value>
    /// <remarks>
    /// <para>The paper clips "the meta-gradient, as well as its norm, at an absolute value of 0.1"; this clips the norm
    /// of each block - feature encoder, encoder, relation network, decoder and each step-size vector. The inner-loop
    /// gradients are never clipped.</para>
    /// </remarks>
    public double? GradientClipThreshold { get; set; } = 0.1;

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
    /// Gets or sets whether to drop the second-order terms of the meta-gradient.
    /// </summary>
    /// <value>Default is false: the exact meta-gradient through every inner step.</value>
    /// <remarks>
    /// <para>Each inner step's gradient is taken on a nested tape that the outer tape differentiates through. True
    /// stops the gradient at each inner step's gradient, as first-order MAML does.</para>
    /// </remarks>
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
    /// Gets or sets the width of each example's feature embedding: the feature encoder's per-example output width.
    /// </summary>
    /// <value>Default is 512. The paper's pre-trained WRN-28-10 features are 640 wide.</value>
    public int EmbeddingDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the width of the relation network's hidden layers.
    /// </summary>
    /// <value>Default is 128, the paper's relation network (Appendix B.3), twice the latent width.</value>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the KL divergence weight for the latent space regularization.
    /// </summary>
    /// <value>Default is 1.33365371e-9, the paper's miniImageNet 1-shot value (Table 6, a random-search value).</value>
    /// <remarks>
    /// <para>Beta in eq. 6: the mean over sampled codes of <c>log q(z) - log p(z)</c> with <c>p = N(0, I)</c>, as the
    /// reference implementation computes it. Evaluation samples nothing and adds no KL term.</para>
    /// <para><b>For Beginners:</b> LEO uses a variational approach where the latent
    /// space is regularized to be close to a prior distribution (usually Gaussian).
    /// This weight controls how strongly we enforce this regularization.
    /// </para>
    /// </remarks>
    public double KLWeight { get; set; } = 1.33365371e-9;

    /// <summary>
    /// Gets or sets the weight of an entropy bonus on the decoder's weight distribution.
    /// </summary>
    /// <value>Default is 0.0 (off - the paper has no entropy term).</value>
    /// <remarks>
    /// <para>An extension: subtracts this weight times the mean log scale of the decoder's Gaussians - their entropy
    /// up to a constant - from the objective, which keeps the generator from collapsing to a point mass. It used to
    /// default to 0.001 and change nothing.</para>
    /// </remarks>
    public double EntropyWeight { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the weight of the penalty pulling the encoder's initial codes toward the adapted codes.
    /// </summary>
    /// <value>Default is 0.124171967, the paper's miniImageNet 1-shot value (Table 6, a random-search value).</value>
    /// <remarks>
    /// <para>Gamma in eq. 6: the mean squared error between stopgrad(adapted codes) and the initial codes.</para>
    /// </remarks>
    public double EncoderPenaltyWeight { get; set; } = 0.124171967;

    /// <summary>
    /// Gets or sets the weight of the decoder orthogonality penalty.
    /// </summary>
    /// <value>Default is 303.216647, the paper's miniImageNet 1-shot value (Table 6, a random-search value).</value>
    /// <remarks>
    /// <para>Lambda 2 in eq. 7: the mean squared deviation from the identity of the correlations between the decoder's
    /// latent rows, as the reference implementation computes it.</para>
    /// </remarks>
    public double OrthogonalityWeight { get; set; } = 303.216647;

    /// <summary>
    /// Gets or sets whether to initialise the decoder with orthonormal latent rows.
    /// </summary>
    /// <value>Default is false (Glorot-uniform, the reference implementation's initializer).</value>
    /// <remarks>
    /// <para>An extension: Gram-Schmidt over Gaussian rows, so the decoder starts where the orthogonality penalty
    /// wants it. It used to rescale blocks of a uniform draw, which is not orthogonal.</para>
    /// </remarks>
    public bool UseOrthogonalInit { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to share the encoder across all classes.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>The paper shares one encoder. False is an extension: one encoder per class slot, applied to the examples
    /// of that class index - useful only where tasks use class indices consistently.</para>
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
    /// <para>The paper's encoder (eq. 3): the relation network sees every ordered pair of support codes and is
    /// averaged per class into that class's Gaussian. Without it, the encoder emits each class Gaussian's parameters
    /// directly and they are averaged over the class's examples.</para>
    /// <para><b>For Beginners:</b> A relation network considers relationships between
    /// all support examples when generating the latent code, which can help when
    /// examples within a class are diverse.
    /// </para>
    /// </remarks>
    public bool UseRelationEncoder { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate on the feature embeddings while meta-training.
    /// </summary>
    /// <value>Default is 1 - 0.711524088, from the paper's miniImageNet 1-shot keep probability (Table 6).</value>
    /// <remarks>
    /// <para>"We applied dropout independently on the feature embedding in every step": a fresh mask for the
    /// encoder, each inner step and the query loss. Adaptation never drops anything. Must be in [0, 1).</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 1.0 - 0.711524088;

    /// <summary>
    /// Gets or sets the L2 weight on the encoder, relation network and decoder.
    /// </summary>
    /// <value>Default is 0.000108982953, the paper's miniImageNet 1-shot value (Table 6, a random-search value).</value>
    /// <remarks>
    /// <para>Lambda 1 in eq. 7, as the reference implementation's l2_regularizer: lambda times half the sum of
    /// squared weights.</para>
    /// </remarks>
    public double L2Regularization { get; set; } = 0.000108982953;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the LEOOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The feature encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public LEOOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
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
               EntropyWeight >= 0 &&
               FineTuningSteps >= 0 &&
               FineTuningLearningRate > 0 &&
               EncoderPenaltyWeight >= 0 &&
               OrthogonalityWeight >= 0 &&
               L2Regularization >= 0 &&
               DropoutRate >= 0 && DropoutRate < 1;
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
            FineTuningSteps = FineTuningSteps,
            FineTuningLearningRate = FineTuningLearningRate,
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
            EncoderPenaltyWeight = EncoderPenaltyWeight,
            OrthogonalityWeight = OrthogonalityWeight,
            UseOrthogonalInit = UseOrthogonalInit,
            ShareEncoder = ShareEncoder,
            UseRelationEncoder = UseRelationEncoder,
            DropoutRate = DropoutRate,
            L2Regularization = L2Regularization
        };
    }

    #endregion
}
