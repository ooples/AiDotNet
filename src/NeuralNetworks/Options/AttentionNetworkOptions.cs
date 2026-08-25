using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the AttentionNetwork.
/// </summary>
/// <remarks>
/// <para>
/// These settings describe the default encoder stack that <see cref="AiDotNet.NeuralNetworks.AttentionNetwork{T}"/>
/// builds when the architecture supplies no explicit layers. The defaults follow Vaswani et al. (2017),
/// "Attention Is All You Need": multi-head self-attention with a position-wise feed-forward network whose
/// inner width is four times the model width.
/// </para>
/// <para><b>For Beginners:</b> These three numbers control how much machinery the network builds for you.
///
/// - <see cref="HeadCount"/> is how many different "points of view" the attention takes at once.
/// - <see cref="EncoderBlockCount"/> is how many times the network re-reads and refines the sequence.
/// - <see cref="FeedForwardExpansion"/> is how much extra room each block gets to think in.
///
/// Larger values give the network more capacity and cost more memory and time.
/// </para>
/// </remarks>
public class AttentionNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the number of attention heads in each encoder block. Default is 8.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multi-head attention splits the embedding into this many independent subspaces, attends within each,
    /// and concatenates the results (Vaswani et al. 2017, §3.2.2). The embedding size must therefore be
    /// divisible by the head count; when it is not, the network uses the largest divisor of the embedding
    /// size that does not exceed this value rather than failing to construct.
    /// </para>
    /// </remarks>
    public int HeadCount { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of stacked transformer encoder blocks. Default is 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each block is <c>LayerNorm(x + SelfAttention(x))</c> followed by <c>LayerNorm(h + FeedForward(h))</c>,
    /// so the residual shortcuts keep a deep stack trainable (Vaswani et al. 2017, §3.1).
    /// </para>
    /// </remarks>
    public int EncoderBlockCount { get; set; } = 3;

    /// <summary>
    /// Gets or sets how much wider the feed-forward inner layer is than the embedding. Default is 4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper uses <c>d_model = 512</c> with <c>d_ff = 2048</c>, a factor of four.
    /// </para>
    /// </remarks>
    public int FeedForwardExpansion { get; set; } = 4;
}
