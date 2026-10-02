namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for Florence-2, Microsoft's unified vision foundation model.
/// </summary>
/// <remarks>
/// <para>
/// Florence-2 (Xiao et al., 2024) is a sequence-to-sequence vision foundation model of 0.23B to 0.77B
/// parameters. It handles captioning, object detection, grounding and OCR through task prompts. A DaViT
/// image encoder feeds projected image tokens, followed by the prompt, to a BART encoder-decoder. The output is
/// generated text whose regions are written as location tokens.
/// </para>
/// <para>
/// The inherited options mean the following here:
/// <list type="bullet">
/// <item><see cref="VisionEncoderOptions.EmbeddingDim"/> is the projected image-token width. It must equal
/// <see cref="TextDim"/>, because image tokens and text share the BART encoder.</item>
/// <item><see cref="VisionEncoderOptions.NumLayers"/> and <see cref="VisionEncoderOptions.NumHeads"/> are
/// the BART encoder's depth and heads.</item>
/// <item><see cref="VisionEncoderOptions.FfnMultiplier"/> is DaViT's MLP ratio.</item>
/// <item><see cref="VisionEncoderOptions.DropoutRate"/> is BART's dropout.</item>
/// <item><see cref="VisionEncoderOptions.PatchSize"/> is DaViT's total stride, 32 (4 x 2 x 2 x 2). The
/// architecture fixes it.</item>
/// </list>
/// </para>
/// </remarks>
public class Florence2Options : VisionEncoderOptions
{
    /// <summary>Creates Florence-2-base options.</summary>
    public Florence2Options() : this(Florence2ModelSize.Base)
    {
    }

    /// <summary>
    /// Creates the published configuration for <paramref name="size"/>. The values come from the
    /// microsoft/Florence-2-base and microsoft/Florence-2-large configurations.
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item>Base: DaViT 128/256/512/1024 wide with 4/8/16/32 heads and groups, a 768-wide projection, and
    /// BART-base text (768 wide, 6 encoder and 6 decoder layers, 12 heads, FFN 3072).</item>
    /// <item>Large: DaViT 256..2048 wide with 8..64 heads, a 1024-wide projection, and BART-large text
    /// (1024 wide, 12 + 12 layers, 16 heads, FFN 4096).</item>
    /// </list>
    /// Both use 768 px pages, depths 1/1/9/1, 12x12 windows, a vocabulary of 51289 and 1000 location bins.
    /// </remarks>
    public Florence2Options(Florence2ModelSize size)
    {
        ModelSize = size;
        bool large = size == Florence2ModelSize.Large;
        ImageSize = 768;
        PatchSize = 32;
        FfnMultiplier = 4;
        DropoutRate = 0.1;
        VisionBaseDim = large ? 256 : 128;
        VisionBaseHeads = large ? 8 : 4;
        EmbeddingDim = large ? 1024 : 768;
        TextDim = large ? 1024 : 768;
        NumLayers = large ? 12 : 6;
        NumDecoderLayers = large ? 12 : 6;
        NumHeads = large ? 16 : 12;
        TextFeedForwardDim = large ? 4096 : 3072;
    }

    /// <summary>Copies every option from <paramref name="other"/>.</summary>
    public Florence2Options(Florence2Options other)
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
        ModelSize = other.ModelSize;
        VisionBaseDim = other.VisionBaseDim;
        VisionBaseHeads = other.VisionBaseHeads;
        VisionThirdStageDepth = other.VisionThirdStageDepth;
        WindowSize = other.WindowSize;
        TextDim = other.TextDim;
        TextFeedForwardDim = other.TextFeedForwardDim;
        NumDecoderLayers = other.NumDecoderLayers;
        VocabSize = other.VocabSize;
        MaxTextPositions = other.MaxTextPositions;
        NumLocationBins = other.NumLocationBins;
        MaxOutputTokens = other.MaxOutputTokens;
        DefaultTask = other.DefaultTask;
    }

    /// <summary>The published size these options were created from. Informational only.</summary>
    public Florence2ModelSize ModelSize { get; }

    /// <summary>DaViT stage-0 width. Widths double at each of the four stages (128 for base, 256 for large).</summary>
    public int VisionBaseDim { get; set; }

    /// <summary>DaViT stage-0 heads, also its channel-attention group count. Doubles per stage (4 or 8).</summary>
    public int VisionBaseHeads { get; set; }

    /// <summary>Depth of DaViT's third stage. The other stages have depth 1. Defaults to 9.</summary>
    public int VisionThirdStageDepth { get; set; } = 9;

    /// <summary>Side of DaViT's spatial attention windows. Defaults to 12.</summary>
    public int WindowSize { get; set; } = 12;

    /// <summary>BART model width, shared by encoder and decoder (768 for base, 1024 for large).</summary>
    public int TextDim { get; set; }

    /// <summary>BART feed-forward width (3072 or 4096).</summary>
    public int TextFeedForwardDim { get; set; }

    /// <summary>Number of BART decoder layers (6 or 12).</summary>
    public int NumDecoderLayers { get; set; }

    /// <summary>Vocabulary size: BART's 50265 BPE tokens plus the added task and location tokens. Defaults to 51289.</summary>
    public int VocabSize { get; set; } = 51289;

    /// <summary>BART's learned positions; bounds both the encoder input and the decoder output. Defaults to 1024.</summary>
    public int MaxTextPositions { get; set; } = 1024;

    /// <summary>
    /// Number of location tokens <c>&lt;loc_0&gt;</c>..<c>&lt;loc_N-1&gt;</c>, each a quantised 0-1 coordinate. Defaults to 1000.
    /// </summary>
    /// <remarks>
    /// This implementation places them at the end of the vocabulary: <c>[VocabSize - NumLocationBins, VocabSize)</c>.
    /// When you load a checkpoint, map its tokenizer's location ids onto that range.
    /// </remarks>
    public int NumLocationBins { get; set; } = 1000;

    /// <summary>Most tokens one generation produces before it stops. Defaults to 1024, the max_new_tokens the reference examples use.</summary>
    public int MaxOutputTokens { get; set; } = 1024;

    /// <summary>The task <c>Predict</c> runs when no task is named. Defaults to OCR.</summary>
    public Florence2Task DefaultTask { get; set; } = Florence2Task.Ocr;

    /// <summary>Throws if the configuration cannot build a Florence-2 model.</summary>
    public void Validate()
    {
        if (ImageSize <= 0 || ImageSize % 32 != 0)
            throw new ArgumentException($"ImageSize ({ImageSize}) must be a positive multiple of DaViT's total stride, 32.", nameof(ImageSize));
        if (PatchSize != 32)
            throw new ArgumentException($"PatchSize is DaViT's total stride and must be 32; got {PatchSize}.", nameof(PatchSize));
        if (ImageSize / 32 > 50)
            throw new ArgumentException($"ImageSize ({ImageSize}) gives a {ImageSize / 32}-wide grid, past the 50 learned image positions.", nameof(ImageSize));
        if (VisionBaseDim <= 0 || VisionBaseHeads <= 0 || VisionBaseDim % VisionBaseHeads != 0)
            throw new ArgumentException($"VisionBaseDim ({VisionBaseDim}) must be a positive multiple of VisionBaseHeads ({VisionBaseHeads}).", nameof(VisionBaseHeads));
        if (VisionThirdStageDepth <= 0) throw new ArgumentException("VisionThirdStageDepth must be positive.", nameof(VisionThirdStageDepth));
        if (WindowSize <= 0) throw new ArgumentException("WindowSize must be positive.", nameof(WindowSize));
        if (FfnMultiplier <= 0) throw new ArgumentException("FfnMultiplier must be positive.", nameof(FfnMultiplier));
        if (EmbeddingDim != TextDim)
            throw new ArgumentException($"EmbeddingDim ({EmbeddingDim}) must equal TextDim ({TextDim}): image tokens and text share the BART encoder.", nameof(EmbeddingDim));
        if (TextDim <= 0 || NumHeads <= 0 || TextDim % NumHeads != 0)
            throw new ArgumentException($"TextDim ({TextDim}) must be a positive multiple of NumHeads ({NumHeads}).", nameof(NumHeads));
        if (TextFeedForwardDim <= 0) throw new ArgumentException("TextFeedForwardDim must be positive.", nameof(TextFeedForwardDim));
        if (NumLayers <= 0 || NumDecoderLayers <= 0) throw new ArgumentException("NumLayers and NumDecoderLayers must be positive.", nameof(NumLayers));
        if (DropoutRate < 0 || DropoutRate >= 1) throw new ArgumentException("DropoutRate must be in [0, 1).", nameof(DropoutRate));
        if (NumLocationBins < 2 || NumLocationBins > VocabSize - 4)
            throw new ArgumentException(
                $"NumLocationBins ({NumLocationBins}) must be at least 2 and leave BOS, PAD, EOS and text tokens in a vocabulary of {VocabSize}.",
                nameof(NumLocationBins));
        int imageTokens = 1 + ((ImageSize / 32) * (ImageSize / 32));
        if (imageTokens + 2 > MaxTextPositions)
            throw new ArgumentException(
                $"{imageTokens} image tokens plus BOS/EOS exceed the {MaxTextPositions} learned text positions.", nameof(MaxTextPositions));
        if (MaxOutputTokens <= 0 || MaxOutputTokens >= MaxTextPositions)
            throw new ArgumentException($"MaxOutputTokens ({MaxOutputTokens}) must be within 1..{MaxTextPositions - 1}.", nameof(MaxOutputTokens));
    }
}
