using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the Conditional Neural Adaptive Processes (CNAP) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// CNAP extends Neural Processes by conditioning on task-specific context points
/// and learning to produce fast adaptation weights for each task. This enables
/// effective few-shot learning through learned task representations.
/// </para>
/// <para>
/// <b>For Beginners:</b> CNAP learns to understand tasks from examples:
/// 1. It encodes examples (context points) into a task representation
/// 2. It uses this representation to generate task-specific fast weights
/// 3. The fast weights modify the base model to work well on that specific task
///
/// This is like having a teacher who can quickly understand what kind of problem
/// you're working on and adjust their teaching style accordingly.
/// </para>
/// <para>
/// <b>Key Components:</b>
/// - <b>Encoder:</b> Processes context points into task representations
/// - <b>Adaptation Network:</b> Generates task-specific fast weights from representations
/// - <b>Base Model:</b> The neural network that gets modified by fast weights
/// </para>
/// </remarks>
public class CNAPOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model is the base network that will be modified by fast weights
    /// during task adaptation. It should be a neural network with parameters
    /// that can be efficiently modulated.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// </summary>
    /// <value>Default: null (uses model's default loss function if available).</value>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter updates (outer loop).
    /// </summary>
    /// <value>Default: null (uses built-in Adam optimizer).</value>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner-loop adaptation.
    /// </summary>
    /// <value>Default: null (uses built-in SGD optimizer).</value>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// </summary>
    /// <value>Default: null (tasks must be provided manually to MetaTrain).</value>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task adaptation).
    /// </summary>
    /// <value>Default: 0.01.</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-update).
    /// </summary>
    /// <value>Default: 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>Default: 1 (CNAP typically uses single-step adaptation with fast weights).</value>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default: 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>Default: 10000.</value>
    public int NumMetaIterations { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default: true (CNAP typically uses first-order for efficiency).</value>
    public bool UseFirstOrder { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default: 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>Default: null (non-deterministic).</value>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    /// <value>Default: 100.</value>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate during meta-training.
    /// </summary>
    /// <value>Default: 500.</value>
    public int EvaluationFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to save checkpoints during training.
    /// </summary>
    /// <value>Default: false.</value>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    /// <value>Default: 1000.</value>
    public int CheckpointFrequency { get; set; } = 1000;

    #endregion

    #region CNAP-Specific Properties

    /// <summary>
    /// Gets or sets the dimension of the task representation vector.
    /// </summary>
    /// <value>Default: 256.</value>
    /// <remarks>
    /// <para>
    /// The representation dimension controls the capacity of the task encoding.
    /// Higher values can capture more complex task relationships but require more
    /// computation and may be prone to overfitting.
    /// </para>
    /// </remarks>
    public int RepresentationDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the hidden dimension for the encoder and adaptation networks.
    /// </summary>
    /// <value>Default: 512.</value>
    public int HiddenDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of layers in the encoder network.
    /// </summary>
    /// <value>Default: 3.</value>
    public int EncoderLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of layers in the adaptation network.
    /// </summary>
    /// <value>Default: 2.</value>
    public int AdaptationNetworkLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to use attention for aggregating context points.
    /// </summary>
    /// <value>Default: true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the encoder uses self-attention to aggregate information
    /// from multiple context points. This allows the model to learn which
    /// context points are most relevant for each task.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Attention helps the model focus on the most
    /// important examples when understanding a new task, rather than treating
    /// all examples equally.
    /// </para>
    /// </remarks>
    public bool UseAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of attention heads when using attention.
    /// </summary>
    /// <value>Default: 4.</value>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Default: 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to normalize fast weights.
    /// </summary>
    /// <value>Default: true.</value>
    /// <remarks>
    /// <para>
    /// Normalizing fast weights helps stabilize training and prevents
    /// the adaptation from making overly large changes to the base model.
    /// </para>
    /// </remarks>
    public bool NormalizeFastWeights { get; set; } = true;

    /// <summary>
    /// Gets or sets the scale factor for fast weights.
    /// </summary>
    /// <value>Default: 1.0.</value>
    public double FastWeightScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the regularization weight for fast weights.
    /// </summary>
    /// <value>Default: 0.001.</value>
    public double FastWeightRegularization { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets how fast weights are applied to the model.
    /// </summary>
    /// <value>Default: Additive.</value>
    /// <remarks>
    /// <para>
    /// - <b>Additive:</b> Fast weights are added to base weights (θ + α)
    /// - <b>Multiplicative:</b> Fast weights scale base weights (θ × α)
    /// - <b>FiLM:</b> Feature-wise Linear Modulation (γθ + β)
    /// </para>
    /// </remarks>
    public FastWeightApplicationMode FastWeightMode { get; set; } = FastWeightApplicationMode.Additive;

    /// <summary>
    /// Gets or sets whether to use layer normalization in networks.
    /// </summary>
    /// <value>Default: true.</value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of context points to use for encoding (support set size).
    /// </summary>
    /// <value>Default: 5 (for 5-shot learning).</value>
    public int ContextSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to predict uncertainty along with predictions.
    /// </summary>
    /// <value>Default: false.</value>
    public bool PredictUncertainty { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for uncertainty loss when predicting uncertainty.
    /// </summary>
    /// <value>Default: 0.1.</value>
    public double UncertaintyWeight { get; set; } = 0.1;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the CNAPOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public CNAPOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all CNAP configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for CNAP training; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps >= 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               RepresentationDimension > 0 &&
               HiddenDimension > 0 &&
               EncoderLayers > 0 &&
               AdaptationNetworkLayers > 0 &&
               ContextSize > 0 &&
               FastWeightScale > 0;
    }

    /// <summary>
    /// Creates a deep copy of the CNAP options.
    /// </summary>
    /// <returns>A new CNAPOptions instance with the same configuration values.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new CNAPOptions<T, TInput, TOutput>(MetaModel)
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
            UseFirstOrder = UseFirstOrder,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            RepresentationDimension = RepresentationDimension,
            HiddenDimension = HiddenDimension,
            EncoderLayers = EncoderLayers,
            AdaptationNetworkLayers = AdaptationNetworkLayers,
            UseAttention = UseAttention,
            NumAttentionHeads = NumAttentionHeads,
            DropoutRate = DropoutRate,
            NormalizeFastWeights = NormalizeFastWeights,
            FastWeightScale = FastWeightScale,
            FastWeightRegularization = FastWeightRegularization,
            FastWeightMode = FastWeightMode,
            UseLayerNorm = UseLayerNorm,
            ContextSize = ContextSize,
            PredictUncertainty = PredictUncertainty,
            UncertaintyWeight = UncertaintyWeight
        };
    }

    #endregion
}

/// <summary>
/// Specifies how fast weights are applied to modify the base model.
/// </summary>
public enum FastWeightApplicationMode
{
    /// <summary>
    /// Fast weights are added to base weights: θ' = θ + α
    /// </summary>
    Additive,

    /// <summary>
    /// Fast weights scale base weights: θ' = θ × (1 + α)
    /// </summary>
    Multiplicative,

    /// <summary>
    /// Feature-wise Linear Modulation: output' = γ × output + β
    /// </summary>
    FiLM
}
