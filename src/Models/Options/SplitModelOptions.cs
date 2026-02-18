namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for split neural network architecture in vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, a neural network is split into parts:
/// each party runs a "bottom model" on its local features, producing an embedding
/// (a compressed representation). The embeddings from all parties are combined and
/// fed into a "top model" that produces the final prediction.</para>
///
/// <para>This class controls how the network is split, how embeddings are sized,
/// and how they're combined.</para>
///
/// <para>Example:
/// <code>
/// var splitOptions = new SplitModelOptions
/// {
///     AggregationMode = VflAggregationMode.Concatenation,
///     SplitPoint = SplitPointStrategy.AutoOptimal,
///     EmbeddingDimension = 64,
///     TopModelHiddenDimension = 128
/// };
/// </code>
/// </para>
/// </remarks>
public class SplitModelOptions
{
    /// <summary>
    /// Gets or sets how embeddings from multiple parties are combined before the top model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Concatenation is the simplest and usually works well.
    /// Use Sum for communication efficiency, Attention for learning party importance.</para>
    /// </remarks>
    public VflAggregationMode AggregationMode { get; set; } = VflAggregationMode.Concatenation;

    /// <summary>
    /// Gets or sets the strategy for choosing where to split the network.
    /// </summary>
    public SplitPointStrategy SplitPoint { get; set; } = SplitPointStrategy.AutoOptimal;

    /// <summary>
    /// Gets or sets the manual split layer index (used only when SplitPoint is Manual).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you choose Manual split, this specifies which layer
    /// separates the bottom model (local) from the top model (coordinator). Layer 0 is the
    /// first hidden layer.</para>
    /// </remarks>
    public int ManualSplitLayerIndex { get; set; }

    /// <summary>
    /// Gets or sets the output dimension of each party's bottom model (embedding size).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each party compresses its features into this many numbers.
    /// Larger dimensions preserve more information but increase communication cost.
    /// A typical value is 32-128.</para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden dimension of the top model. The top model processes the
    /// combined embeddings to produce the final prediction.
    /// </summary>
    public int TopModelHiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of hidden layers in the top model.
    /// </summary>
    public int TopModelHiddenLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to use batch normalization in the bottom models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batch normalization helps stabilize training. However,
    /// it requires computing statistics across the batch, which may leak information
    /// about the local data distribution. Disable for maximum privacy.</para>
    /// </remarks>
    public bool UseBatchNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to add gradient noise to embeddings before sending
    /// them to the coordinator, reducing information leakage.
    /// </summary>
    public bool AddEmbeddingNoise { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of Gaussian noise added to embeddings
    /// when <see cref="AddEmbeddingNoise"/> is enabled.
    /// </summary>
    public double EmbeddingNoiseScale { get; set; } = 0.01;
}
