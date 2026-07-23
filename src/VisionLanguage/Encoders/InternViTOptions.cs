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
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public InternViTOptions(InternViTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        EmbeddingDim = other.EmbeddingDim;
        PatchSize = other.PatchSize;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FfnMultiplier = other.FfnMultiplier;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxTiles = other.MaxTiles;
        PixelShuffleRatio = other.PixelShuffleRatio;
        UseDynamicResolution = other.UseDynamicResolution;
        Use3DRoPE = other.Use3DRoPE;
    }

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
