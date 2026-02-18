namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for InternViT, the vision encoder used in the InternVL series.
/// </summary>
/// <remarks>
/// <para>InternViT (Chen et al., 2024) is a 6B-parameter ViT designed for progressive alignment
/// with LLMs. It uses dynamic resolution processing and pixel shuffle downsampling to handle
/// images of varying sizes efficiently as part of the InternVL architecture.</para>
/// </remarks>
public class InternViTOptions : VisionEncoderOptions
{
    /// <summary>
    /// Gets or sets the maximum number of tiles for dynamic resolution.
    /// </summary>
    public int MaxTiles { get; set; } = 12;

    /// <summary>
    /// Gets or sets the pixel shuffle downsampling ratio.
    /// </summary>
    public int PixelShuffleRatio { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to use dynamic resolution tiling.
    /// </summary>
    public bool UseDynamicResolution { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use 3D-RoPE for positional encoding.
    /// </summary>
    public bool Use3DRoPE { get; set; } = false;

    public InternViTOptions()
    {
        ImageSize = 448;
        EmbeddingDim = 3200;
        PatchSize = 14;
        NumLayers = 48;
        NumHeads = 25;
    }
}
