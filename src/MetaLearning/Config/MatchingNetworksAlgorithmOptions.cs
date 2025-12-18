using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for the Matching Networks algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Matching Networks use attention mechanisms over the support set to classify
/// query examples. This configuration controls how the attention mechanism works
/// and how examples are encoded.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how Matching Networks learns to compare examples:
///
/// Key parameters:
/// - <b>AttentionFunction:</b> How to measure similarity (Cosine, DotProduct, Learned)
/// - <b>UseFullContext:</b> Whether to use all examples when encoding each example
/// - <b>UseBidirectionalEncoding:</b> Whether to process sequences bidirectionally
/// - <b>L2Regularization:</b> Regularization strength for encoder
/// - <b>Temperature:</b> Controls sharpness of attention distribution
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Bidirectional LSTM encoding for sequences
/// - Set-to-set attention mechanisms
/// - Learnable similarity functions
/// - Cached embeddings for fast inference
/// - Support for both classification and regression
/// </para>
/// </remarks>
public class MatchingNetworksAlgorithmOptions<T, TInput, TOutput> 
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the neural network used as encoder.
    /// </summary>
    /// <value>
    /// The encoder that embeds inputs into a comparable space.
    /// This network learns to produce embeddings suitable for attention-based classification.
    /// </value>
    /// <remarks>
    /// The encoder can be any neural network architecture:
    /// - CNN for images
    /// - RNN/LSTM for sequences
    /// - Transformer for text
    /// - MLP for tabular data
    /// </remarks>
    public INeuralNetwork<T>? Encoder { get; set; }

    /// <summary>
    /// Gets or sets the attention function used for computing similarity.
    /// </summary>
    /// <value>
    /// The function used to compute attention weights between examples.
    /// Default is Cosine.
    /// </value>
    /// <remarks>
    /// <b>Attention Functions:</b>
    /// - <b>Cosine:</b> Cosine similarity (normalized, works well)
    /// - <b>DotProduct:</b> Raw dot product (sensitive to magnitude)
    /// - <b>Euclidean:</b> Negative Euclidean distance
    /// - <b>Learned:</b> Small neural network learns similarity (most flexible)
    /// </remarks>
    public AttentionFunction AttentionType { get; set; } = AttentionFunction.Cosine;

    /// <summary>
    /// Gets or sets whether to use full context encoding.
    /// </summary>
    /// <value>
    /// If true, encodes each example using attention over all other examples.
    /// This enables the network to learn task-specific representations.
    /// Default is true.
    /// </value>
    /// <remarks>
    /// Full context encoding allows each example's embedding to be influenced
    /// by all other examples in the episode, making the representations task-aware.
    /// This is a key innovation of Matching Networks.
    /// </remarks>
    public bool UseFullContext { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use bidirectional encoding for sequences.
    /// </summary>
    /// <value>
    /// If true, processes sequences bidirectionally (forward and backward).
    /// Only applicable to sequence data (text, time series).
    /// Default is false.
    /// </value>
    /// <remarks>
    /// Bidirectional encoding captures context from both directions,
    /// which is crucial for understanding sequences in context.
    /// </remarks>
    public bool UseBidirectionalEncoding { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of classes in the classification task.
    /// </summary>
    /// <value>
    /// Number of output classes (N in N-way classification).
    /// This determines the size of the output layer.
    /// Default is 5.
    /// </value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the temperature for attention softmax.
    /// </summary>
    /// <value>
    /// Temperature value controlling the sharpness of the attention distribution.
    /// Lower values make attention more peaked (focus on few examples).
    /// Default is 1.0.
    /// </value>
    /// <remarks>
    /// Temperature affects how much attention is spread across support examples:
    /// - < 1.0: More focused attention
    /// - 1.0: Standard softmax
    /// - > 1.0: More uniform attention
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the L2 regularization coefficient.
    /// </summary>
    /// <value>
    /// Regularization strength for encoder parameters.
    /// Helps prevent overfitting in few-shot scenarios.
    /// Default is 1e-4.
    /// </value>
    public double L2Regularization { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the dropout rate for the encoder.
    /// </summary>
    /// <value>
    /// Dropout rate between 0.0 and 1.0.
    /// Applied to encoder layers during training.
    /// Default is 0.1.
    /// </value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to cache support embeddings during inference.
    /// </summary>
    /// <value>
    /// If true, caches support set embeddings for faster inference.
    /// Uses more memory but speeds up predictions.
    /// Default is true.
    /// </value>
    public bool CacheSupportEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use label smoothing during training.
    /// </summary>
    /// <value>
    /// If true, applies label smoothing to prevent overconfidence.
    /// Can improve generalization in few-shot learning.
    /// Default is false.
    /// </value>
    public bool UseLabelSmoothing { get; set; } = false;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    /// <value>
    /// Amount of smoothing to apply (0.0 to 1.0).
    /// Only used when UseLabelSmoothing is true.
    /// Default is 0.1.
    /// </value>
    public double LabelSmoothingFactor { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use memory-efficient attention.
    /// </summary>
    /// <value>
    /// If true, uses memory-efficient attention computation.
    /// Trades computation for memory usage.
    /// Default is false.
    /// </value>
    public bool UseMemoryEfficientAttention { get; set; } = false;

    /// <summary>
    /// Gets or sets the chunk size for memory-efficient attention.
    /// </summary>
    /// <value>
    /// Size of chunks when using memory-efficient attention.
    /// Only used when UseMemoryEfficientAttention is true.
    /// Default is 32.
    /// </value>
    public int AttentionChunkSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to normalize embeddings.
    /// </summary>
    /// <value>
    /// If true, applies L2 normalization to embeddings.
    /// Can improve stability of cosine similarity.
    /// Default is true.
    /// </value>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use attention temperature scheduling.
    /// </summary>
    /// <value>
    /// If true, anneals temperature during training.
    /// Starts with high temperature, ends with low temperature.
    /// Default is false.
    /// </value>
    public bool UseTemperatureScheduling { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial temperature for scheduling.
    /// </summary>
    /// <value>
    /// Starting temperature when using temperature scheduling.
    /// Default is 10.0.
    /// </value>
    public double InitialTemperature { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the final temperature for scheduling.
    /// </summary>
    /// <value>
    /// Ending temperature when using temperature scheduling.
    /// Default is 1.0.
    /// </value>
    public double FinalTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of episodes for temperature annealing.
    /// </summary>
    /// <value>
    /// Number of training episodes over which to anneal temperature.
    /// Default is 10000.
    /// </value>
    public int TemperatureAnnealingEpisodes { get; set; } = 10000;

    /// <summary>
    /// Creates a default Matching Networks configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on the original Matching Networks paper:
    /// - Attention function: Cosine similarity
    /// - Full context encoding enabled
    /// - L2 regularization with coefficient 1e-4
    /// - Normalized embeddings for stability
    /// </remarks>
    public MatchingNetworksAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 1; // Matching Networks don't need inner loop
    }

    /// <summary>
    /// Creates a Matching Networks configuration with custom values.
    /// </summary>
    /// <param name="encoder">The neural network encoder.</param>
    /// <param name="attentionFunction">The attention function to use.</param>
    /// <param name="numClasses">Number of classes for classification.</param>
    /// <param name="useFullContext">Whether to use full context encoding.</param>
    /// <param name="useBidirectionalEncoding">Whether to use bidirectional encoding.</param>
    /// <param name="temperature">Temperature for attention softmax.</param>
    /// <param name="l2Regularization">L2 regularization coefficient.</param>
    /// <param name="dropoutRate">Dropout rate for the encoder.</param>
    /// <param name="innerLearningRate">Learning rate for optimization.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public MatchingNetworksAlgorithmOptions(
        INeuralNetwork<T> encoder,
        AttentionFunction attentionFunction = AttentionFunction.Cosine,
        int numClasses = 5,
        bool useFullContext = true,
        bool useBidirectionalEncoding = false,
        double temperature = 1.0,
        double l2Regularization = 1e-4,
        double dropoutRate = 0.1,
        double innerLearningRate = 0.001,
        int numEpisodes = 10000)
    {
        Encoder = encoder;
        AttentionType = attentionFunction;
        NumClasses = numClasses;
        UseFullContext = useFullContext;
        UseBidirectionalEncoding = useBidirectionalEncoding;
        Temperature = temperature;
        L2Regularization = l2Regularization;
        DropoutRate = dropoutRate;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = 1; // Matching Networks don't use inner loop
        NumEpisodes = numEpisodes;
    }

    /// <summary>
    /// Validates the configuration parameters.
    /// </summary>
    /// <returns>True if all parameters are valid, false otherwise.</returns>
    public virtual bool IsValid()
    {
        // Check base class validation
            return false;

        // Check encoder
        if (Encoder == null)
            return false;

        // Check number of classes
        if (NumClasses <= 1 || NumClasses > 1000)
            return false;

        // Check temperature
        if (Temperature <= 0.0 || Temperature > 100.0)
            return false;

        // Check regularization parameters
        if (L2Regularization < 0.0 || L2Regularization > 1.0)
            return false;

        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            return false;

        // Check label smoothing
        if (UseLabelSmoothing)
        {
            if (LabelSmoothingFactor < 0.0 || LabelSmoothingFactor > 1.0)
                return false;
        }

        // Check attention chunk size
        if (UseMemoryEfficientAttention && AttentionChunkSize <= 0)
            return false;

        // Check temperature scheduling
        if (UseTemperatureScheduling)
        {
            if (InitialTemperature <= 0.0 || FinalTemperature <= 0.0)
                return false;
            if (InitialTemperature <= FinalTemperature)
                return false; // Should start high and go low
            if (TemperatureAnnealingEpisodes <= 0)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets the current temperature for the given training episode.
    /// </summary>
    /// <param name="currentEpisode">Current episode number.</param>
    /// <returns>Temperature value for this episode.</returns>
    public double GetCurrentTemperature(int currentEpisode)
    {
        if (!UseTemperatureScheduling)
        {
            return Temperature;
        }

        if (currentEpisode >= TemperatureAnnealingEpisodes)
        {
            return FinalTemperature;
        }

        // Linear interpolation
        double progress = (double)currentEpisode / TemperatureAnnealingEpisodes;
        return InitialTemperature - progress * (InitialTemperature - FinalTemperature);
    }
}