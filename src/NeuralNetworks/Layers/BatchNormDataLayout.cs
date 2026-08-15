namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// How <see cref="BatchNormalizationLayer{T}"/> should interpret a rank-3 input.
/// </summary>
/// <remarks>
/// <para>
/// Rank 3 is ambiguous. <c>[C, H, W]</c> is an unbatched image with channels on axis 0;
/// <c>[B, C, T]</c> is a batched channels-first sequence with channels on axis 1. Nothing in the
/// shape distinguishes them, so the layer used to decide by testing whether the trailing axis
/// happened to equal the parameter count -- a coincidence, not a fact, and one that silently
/// transposed ContextNet's statistics when T equalled C.
/// </para>
/// <para><b>For Beginners:</b> "channels" are the feature maps a convolution produces. Batch norm
/// has to normalize each channel separately, so it needs to know which position in the shape the
/// channels live in. For a batch of audio the order is usually [batch, channels, time]; for a
/// single image it is [channels, height, width]. Both are three numbers, so the layer cannot tell
/// which you meant -- this lets you say.</para>
/// </remarks>
public enum BatchNormDataLayout
{
    /// <summary>
    /// Historical behaviour: treat rank 3 as unbatched channels-first <c>[C, H, W]</c>, and take
    /// the features-last flatten when the trailing axis equals the parameter count. The default, so
    /// no existing caller changes.
    /// </summary>
    Infer = 0,

    /// <summary>
    /// Channels are on axis 1 of a batched tensor: <c>[B, C, ...]</c>. Suppresses the features-last
    /// flatten regardless of what the trailing axis happens to equal.
    /// </summary>
    ChannelsFirst,

    /// <summary>
    /// Channels are on the trailing axis: <c>[B, ..., C]</c>.
    /// </summary>
    ChannelsLast,
}
