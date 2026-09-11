namespace AiDotNet.Video.ActionRecognition;

/// <summary>
/// Index map of the default VideoMAE layer stack built by
/// <see cref="AiDotNet.Helpers.LayerHelper{T}.CreateDefaultVideoMAELayers"/>.
/// </summary>
/// <remarks>
/// <para>
/// VideoMAE does not run its <c>Layers</c> list as one sequential chain: the classification forward, the
/// pretraining encoder and the reconstruction decoder each pick out their own slice of it. Those slices
/// used to be addressed with bare numbers in the model while the factory built the list from loop counts,
/// so the two could drift apart silently — and did: the decoder was addressed as starting at index 15,
/// which is the classifier's <c>DenseLayer</c>, so pretraining ran the encoder features through the class
/// head and never reached the reconstruction head. Both the factory and the model now read the layout
/// from here.
/// </para>
/// <para>
/// Layout: <c>[patch-embed | encoder blocks | feature-reduce conv, global pool, classifier |
/// decoder blocks | reconstruction head]</c>.
/// </para>
/// </remarks>
internal static class VideoMAELayerLayout
{
    /// <summary>Number of encoder blocks in the default stack.</summary>
    public const int EncoderBlockCount = 12;

    /// <summary>Number of reconstruction-decoder blocks in the default stack.</summary>
    public const int DecoderBlockCount = 4;

    /// <summary>The tubelet patch-embedding convolution.</summary>
    public const int PatchEmbedIndex = 0;

    /// <summary>First encoder block.</summary>
    public const int FirstEncoderBlockIndex = PatchEmbedIndex + 1;

    /// <summary>One past the last encoder block.</summary>
    public const int EncoderEndIndex = FirstEncoderBlockIndex + EncoderBlockCount;

    /// <summary>The 1x1 feature-reduce convolution at the start of the classification head.</summary>
    public const int FeatureReduceIndex = EncoderEndIndex;

    /// <summary>The classification head's global pooling layer (not on the model's forward path; the
    /// model pools per tubelet itself).</summary>
    public const int GlobalPoolIndex = FeatureReduceIndex + 1;

    /// <summary>The final classification <c>DenseLayer</c> (raw logits).</summary>
    public const int ClassifierIndex = GlobalPoolIndex + 1;

    /// <summary>First reconstruction-decoder block.</summary>
    public const int FirstDecoderBlockIndex = ClassifierIndex + 1;

    /// <summary>The reconstruction head, which predicts every pixel of a tubelet patch.</summary>
    public const int ReconstructionHeadIndex = FirstDecoderBlockIndex + DecoderBlockCount;

    /// <summary>Total number of layers in the default stack.</summary>
    public const int LayerCount = ReconstructionHeadIndex + 1;
}
