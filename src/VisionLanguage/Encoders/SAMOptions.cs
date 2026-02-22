namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the Segment Anything Model (SAM) vision encoder.
/// </summary>
/// <remarks>
/// <para>SAM (Kirillov et al., 2023) consists of a ViT-based image encoder that produces image embeddings,
/// which can be combined with prompt embeddings (points, boxes, masks) via a lightweight mask decoder
/// for promptable segmentation of any object.</para>
/// </remarks>
public class SAMOptions : VisionEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAMOptions(SAMOptions other)
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
        MaxPointsPerPrompt = other.MaxPointsPerPrompt;
        MaskDecoderDim = other.MaskDecoderDim;
        NumMaskDecoderLayers = other.NumMaskDecoderLayers;
        NumMultimaskOutputs = other.NumMultimaskOutputs;
        UseRelativePositionalEncoding = other.UseRelativePositionalEncoding;
        WindowSize = other.WindowSize;
    }

    /// <summary>
    /// Gets or sets the maximum number of points per prompt.
    /// </summary>
    public int MaxPointsPerPrompt { get; set; } = 16;

    /// <summary>
    /// Gets or sets the mask decoder embedding dimension.
    /// </summary>
    public int MaskDecoderDim { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of mask decoder layers.
    /// </summary>
    public int NumMaskDecoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of multimask outputs.
    /// </summary>
    public int NumMultimaskOutputs { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to use relative positional encoding (window attention).
    /// </summary>
    public bool UseRelativePositionalEncoding { get; set; } = true;

    /// <summary>
    /// Gets or sets the window size for windowed attention in the encoder.
    /// </summary>
    public int WindowSize { get; set; } = 14;

    public SAMOptions()
    {
        ImageSize = 1024;
        EmbeddingDim = 768;
        PatchSize = 16;
        NumLayers = 12;
        NumHeads = 12;
    }
}
