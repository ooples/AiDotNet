namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Turns the shared <see cref="UNetNoisePredictor{T}"/> into MVDream's multi-view U-Net (Shi et al., 2023, arXiv
/// 2308.16512; official <c>bytedance/MVDream</c>, <c>MultiViewUNetModel</c>) and, with image-prompt tokens, into
/// ImageDream's (Wang and Shi, 2023, arXiv 2312.02201; official <c>bytedance/ImageDream</c>).
/// </summary>
/// <remarks>
/// <para>
/// With these options every attention position of the U-Net holds an LDM <c>SpatialTransformer3D</c> instead of the
/// default self- and cross-attention pair: GroupNorm(32, eps 1e-6), a linear <c>proj_in</c>, then per transformer
/// block a LayerNorm and 3D self-attention over the tokens of all the object's views together, a LayerNorm and
/// cross-attention per view, a LayerNorm and a GEGLU feed-forward, then a zero-initialized linear <c>proj_out</c> and
/// the input added back (<c>mvdream/ldm/modules/attention.py</c>, <c>use_linear_in_transformer: True</c>).
/// </para>
/// <para>
/// Leave the predictor's options null and nothing changes: the default U-Net builds and computes exactly as before.
/// </para>
/// <para><b>For Beginners:</b> the U-Net normally looks at one picture at a time. MVDream lets every pixel of every
/// view look at every pixel of the other views of the same object, which is what keeps the views consistent.</para>
/// </remarks>
public sealed class MultiViewUNetOptions
{
    /// <summary>
    /// The number of consecutive batch items that are views of one object (MVDream's <c>num_frames</c>); the 3D
    /// self-attention runs over the tokens of each such group. Zero (the default) treats each forward call's whole
    /// batch as the views of one object, which is how <c>MultiViewUNet</c> calls it (the conditional and unconditional
    /// branches of guidance run as separate calls).
    /// </summary>
    public int FramesPerObject { get; set; } = 0;

    /// <summary>The width of each attention head (LDM <c>num_head_channels</c>; 64 in MVDream's sd-v2-base config).</summary>
    /// <remarks>A transformer has <c>channels / HeadChannels</c> heads. A width below one head (or one that does not
    /// split into heads of this width), which only test-sized U-Nets have, uses a single head.</remarks>
    public int HeadChannels { get; set; } = 64;

    /// <summary>
    /// A fixed number of attention heads per transformer (LDM <c>num_heads</c>, 8 in Stable Diffusion 1.x, whose heads
    /// are <c>channels / 8</c> wide). Zero (the default) uses <see cref="HeadChannels"/>-wide heads instead
    /// (<c>num_head_channels</c>, as Stable Diffusion 2.x and MVDream do).
    /// </summary>
    public int HeadCount { get; set; } = 0;

    /// <summary>
    /// The transformer's <c>proj_in</c>/<c>proj_out</c>: linear (<c>use_linear_in_transformer: True</c>, SD 2.x and
    /// MVDream) or 1x1 convolutions (SD 1.x). Both compute the same per-position map with the same parameters; the
    /// choice sets the checkpoint shapes the loader reads (<c>[C, C]</c> or <c>[C, C, 1, 1]</c>).
    /// </summary>
    public SpatialTransformerProjection Projection { get; set; } = SpatialTransformerProjection.Linear;

    /// <summary>The number of transformer blocks per spatial transformer (LDM <c>transformer_depth</c>; 1).</summary>
    public int TransformerDepth { get; set; } = 1;

    /// <summary>
    /// ImageDream's image-prompt token count (<c>ip_dim</c>; 16 in <c>sd_v2_base_ipmv_local.yaml</c>). When positive,
    /// every transformer's cross-attention is IP-Adapter-style decoupled attention with its own <c>to_k_ip</c> and
    /// <c>to_v_ip</c>, and the last this-many context tokens are the image tokens. Zero (the default) is MVDream's
    /// text-only cross-attention.
    /// </summary>
    public int ImagePromptTokens { get; set; } = 0;

    /// <summary>ImageDream's <c>ip_weight</c>: the scale of the image attention added to the text attention (1.0).</summary>
    public double ImagePromptWeight { get; set; } = 1.0;

    /// <summary>The width of the CLIP image features ImageDream's resampler reads (<c>embedding_dim</c>; 1280, the
    /// width of OpenCLIP ViT-H/14's penultimate hidden states).</summary>
    public int ImageFeatureDim { get; set; } = 1280;

    /// <summary>The resampler's number of perceiver layers (<c>depth</c>; 4).</summary>
    public int ResamplerDepth { get; set; } = 4;

    /// <summary>The resampler's number of attention heads (<c>heads</c>; 12).</summary>
    public int ResamplerHeads { get; set; } = 12;

    /// <summary>The resampler's head width (<c>dim_head</c>; 64).</summary>
    public int ResamplerHeadDim { get; set; } = 64;

    /// <summary>The resampler's feed-forward expansion (<c>ff_mult</c>; 4).</summary>
    public int ResamplerFeedForwardMultiplier { get; set; } = 4;

    /// <summary>
    /// Stable Diffusion 1.x's <c>SpatialTransformer</c> (<c>v1-inference.yaml</c>): one view per object (plain 2D
    /// self-attention), 8 heads, 1x1-convolution projections. The U-Net is then LDM's <c>UNetModel</c> exactly.
    /// </summary>
    public static MultiViewUNetOptions StableDiffusion1() => new MultiViewUNetOptions
    {
        FramesPerObject = 1,
        HeadCount = 8,
        Projection = SpatialTransformerProjection.Convolution
    };

    /// <summary>
    /// Stable Diffusion 2.x's <c>SpatialTransformer</c> (<c>v2-inference.yaml</c>): one view per object, 64-wide heads,
    /// linear projections.
    /// </summary>
    public static MultiViewUNetOptions StableDiffusion2() => new MultiViewUNetOptions { FramesPerObject = 1 };

    /// <summary>MVDream's text-to-multi-view configuration: 3D self-attention, text cross-attention.</summary>
    public static MultiViewUNetOptions MVDream() => new MultiViewUNetOptions();

    /// <summary>ImageDream's configuration: MVDream plus decoupled image cross-attention over 16 image tokens.</summary>
    public static MultiViewUNetOptions ImageDream() => new MultiViewUNetOptions { ImagePromptTokens = 16, ImagePromptWeight = 1.0 };

    internal void Validate()
    {
        if (FramesPerObject < 0) throw new ArgumentOutOfRangeException(nameof(FramesPerObject), "FramesPerObject cannot be negative.");
        if (HeadCount < 0) throw new ArgumentOutOfRangeException(nameof(HeadCount), "HeadCount cannot be negative.");
        if (HeadChannels <= 0) throw new ArgumentOutOfRangeException(nameof(HeadChannels), "HeadChannels must be positive.");
        if (TransformerDepth <= 0) throw new ArgumentOutOfRangeException(nameof(TransformerDepth), "TransformerDepth must be positive.");
        if (ImagePromptTokens < 0) throw new ArgumentOutOfRangeException(nameof(ImagePromptTokens), "ImagePromptTokens cannot be negative.");
        if (ImageFeatureDim <= 0 || ResamplerDepth <= 0 || ResamplerHeads <= 0 || ResamplerHeadDim <= 0 || ResamplerFeedForwardMultiplier <= 0)
            throw new ArgumentOutOfRangeException(nameof(ImageFeatureDim), "The resampler sizes must be positive.");
        if (double.IsNaN(ImagePromptWeight) || double.IsInfinity(ImagePromptWeight))
            throw new ArgumentOutOfRangeException(nameof(ImagePromptWeight), "ImagePromptWeight must be finite.");
    }

    /// <summary>The head count of a transformer over <paramref name="channels"/> channels.</summary>
    internal int HeadsFor(int channels)
    {
        if (HeadCount > 0) return channels % HeadCount == 0 ? HeadCount : 1;
        int heads = channels / HeadChannels;
        return heads >= 1 && channels % HeadChannels == 0 ? heads : 1;
    }
}
