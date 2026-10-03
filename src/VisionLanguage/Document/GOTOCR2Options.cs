namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for GOT-OCR2, the General OCR Theory model (Wei et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// GOT-OCR2 has three parts:
/// <list type="bullet">
/// <item>A SAM ViTDet-B image encoder over 1024 px pages: patch 16, 768 wide, 12 blocks, windowed 14x14
/// attention with global attention in blocks 2/5/8/11, and a 256-channel neck.</item>
/// <item>Two stride-2 convolutions that reduce the page to 256 image tokens.</item>
/// <item>The Qwen-0.5B decoder: 1024 wide, 24 layers, 16 heads with 16 KV heads, a SiLU-gated FFN of 2816,
/// RoPE theta 1e6, and a 151860-token vocabulary.</item>
/// </list>
/// These are the defaults of HF <c>GotOcr2Config</c>.
/// </para>
/// <para>
/// The inherited options mean the following here: <see cref="DocumentVLMOptions.VisionDim"/>,
/// <see cref="DocumentVLMOptions.NumVisionLayers"/> and <see cref="DocumentVLMOptions.NumHeads"/> describe the
/// ViTDet encoder. <see cref="DocumentVLMOptions.DecoderDim"/> and
/// <see cref="DocumentVLMOptions.NumDecoderLayers"/> describe the Qwen decoder.
/// <see cref="DocumentVLMOptions.MaxSequenceLength"/> bounds the full prompt plus answer.
/// </para>
/// </remarks>
public class GOTOCR2Options : DocumentVLMOptions
{
    /// <summary>Copies every option from <paramref name="other"/>.</summary>
    public GOTOCR2Options(GOTOCR2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        IsOcrFree = other.IsOcrFree;
        MaxPages = other.MaxPages;
        MaxOutputTokens = other.MaxOutputTokens;
        PatchSize = other.PatchSize;
        VisionMlpDim = other.VisionMlpDim;
        WindowSize = other.WindowSize;
        GlobalAttentionEvery = other.GlobalAttentionEvery;
        NeckChannels = other.NeckChannels;
        DecoderHeads = other.DecoderHeads;
        DecoderKeyValueHeads = other.DecoderKeyValueHeads;
        DecoderFeedForwardDim = other.DecoderFeedForwardDim;
        RopeTheta = other.RopeTheta;
        RmsNormEpsilon = other.RmsNormEpsilon;
        EndOfTextTokenId = other.EndOfTextTokenId;
        ImStartTokenId = other.ImStartTokenId;
        ImEndTokenId = other.ImEndTokenId;
        FormatOutput = other.FormatOutput;
    }

    /// <summary>Creates the published GOT-OCR2 configuration.</summary>
    public GOTOCR2Options()
    {
        ImageSize = 1024;
        VisionDim = 768;
        NumVisionLayers = 12;
        NumHeads = 12;
        DecoderDim = 1024;
        NumDecoderLayers = 24;
        VocabSize = 151860;
        MaxSequenceLength = 8192;
        MaxGenerationLength = 4096;
    }

    /// <summary>ViTDet patch size, also the encoder's stride. Defaults to 16.</summary>
    public int PatchSize { get; set; } = 16;

    /// <summary>ViTDet MLP width. Defaults to 3072.</summary>
    public int VisionMlpDim { get; set; } = 3072;

    /// <summary>Side of ViTDet's windowed-attention windows. Defaults to 14.</summary>
    public int WindowSize { get; set; } = 14;

    /// <summary>
    /// Every N-th ViTDet block attends globally; the rest use windows. Defaults to 3, which gives blocks 2, 5, 8
    /// and 11 of 12.
    /// </summary>
    public int GlobalAttentionEvery { get; set; } = 3;

    /// <summary>Channels of ViTDet's neck output. Defaults to 256.</summary>
    public int NeckChannels { get; set; } = 256;

    /// <summary>Qwen decoder query heads. Defaults to 16.</summary>
    public int DecoderHeads { get; set; } = 16;

    /// <summary>Qwen decoder key/value heads. Defaults to 16: Qwen-0.5B does not group its keys and values.</summary>
    public int DecoderKeyValueHeads { get; set; } = 16;

    /// <summary>Qwen decoder SiLU-gated feed-forward width. Defaults to 2816.</summary>
    public int DecoderFeedForwardDim { get; set; } = 2816;

    /// <summary>Rotary embedding base. Defaults to 1,000,000.</summary>
    public double RopeTheta { get; set; } = 1_000_000.0;

    /// <summary>RMSNorm epsilon. Defaults to 1e-6.</summary>
    public double RmsNormEpsilon { get; set; } = 1e-6;

    /// <summary>Qwen's <c>&lt;|endoftext|&gt;</c> id. Defaults to 151643.</summary>
    public int EndOfTextTokenId { get; set; } = 151643;

    /// <summary>ChatML's <c>&lt;|im_start|&gt;</c> id. Defaults to 151644.</summary>
    public int ImStartTokenId { get; set; } = 151644;

    /// <summary>ChatML's <c>&lt;|im_end|&gt;</c> id, which also ends an answer. Defaults to 151645.</summary>
    public int ImEndTokenId { get; set; } = 151645;

    /// <summary>
    /// When true, the query is "OCR with format: ", which asks for formatted output (Markdown, LaTeX, music
    /// notation). Otherwise it is "OCR: " for plain text. Defaults to false, the processor's default.
    /// </summary>
    public bool FormatOutput { get; set; }

    /// <summary>
    /// <c>&lt;imgpad&gt;</c>, the image-slot token: the last id of the vocabulary. <c>&lt;/img&gt;</c> and
    /// <c>&lt;img&gt;</c> are the two ids before it, as in the GOT-OCR2 tokenizer (151859, 151858, 151857).
    /// </summary>
    public int ImagePadTokenId => VocabSize - 1;

    /// <summary>The <c>&lt;/img&gt;</c> id.</summary>
    public int ImageEndTokenId => VocabSize - 2;

    /// <summary>The <c>&lt;img&gt;</c> id.</summary>
    public int ImageStartTokenId => VocabSize - 3;

    /// <summary>Throws if the configuration cannot build GOT-OCR2.</summary>
    public void Validate()
    {
        if (PatchSize <= 0 || ImageSize <= 0 || ImageSize % PatchSize != 0)
            throw new ArgumentException($"ImageSize ({ImageSize}) must be a positive multiple of PatchSize ({PatchSize}).", nameof(ImageSize));
        if ((ImageSize / PatchSize) % 4 != 0)
            throw new ArgumentException($"The {ImageSize / PatchSize}-wide patch grid must be a multiple of 4 for the projector's two stride-2 convolutions.", nameof(ImageSize));
        if (VisionDim <= 0 || NumHeads <= 0 || VisionDim % NumHeads != 0)
            throw new ArgumentException($"VisionDim ({VisionDim}) must be a positive multiple of NumHeads ({NumHeads}).", nameof(NumHeads));
        if (NumVisionLayers <= 0 || VisionMlpDim <= 0 || WindowSize <= 0 || NeckChannels <= 0)
            throw new ArgumentException("NumVisionLayers, VisionMlpDim, WindowSize and NeckChannels must be positive.", nameof(NumVisionLayers));
        if (GlobalAttentionEvery < 2 || GlobalAttentionEvery > NumVisionLayers)
            throw new ArgumentException($"GlobalAttentionEvery ({GlobalAttentionEvery}) must be within 2..NumVisionLayers ({NumVisionLayers}).", nameof(GlobalAttentionEvery));
        if (DecoderDim <= 0 || DecoderHeads <= 0 || DecoderDim % DecoderHeads != 0 || (DecoderDim / DecoderHeads) % 2 != 0)
            throw new ArgumentException($"DecoderDim ({DecoderDim}) must split into an even head width over DecoderHeads ({DecoderHeads}).", nameof(DecoderHeads));
        if (DecoderKeyValueHeads <= 0 || DecoderHeads % DecoderKeyValueHeads != 0)
            throw new ArgumentException($"DecoderHeads ({DecoderHeads}) must be a multiple of DecoderKeyValueHeads ({DecoderKeyValueHeads}).", nameof(DecoderKeyValueHeads));
        if (NumDecoderLayers <= 0 || DecoderFeedForwardDim <= 0)
            throw new ArgumentException("NumDecoderLayers and DecoderFeedForwardDim must be positive.", nameof(NumDecoderLayers));
        if (RopeTheta <= 0 || RmsNormEpsilon <= 0)
            throw new ArgumentException("RopeTheta and RmsNormEpsilon must be positive.", nameof(RopeTheta));
        foreach (var (id, name) in new[] { (EndOfTextTokenId, nameof(EndOfTextTokenId)), (ImStartTokenId, nameof(ImStartTokenId)), (ImEndTokenId, nameof(ImEndTokenId)) })
            if (id < 0 || id >= ImageStartTokenId)
                throw new ArgumentException($"{name} ({id}) must be a vocabulary id below the three image tokens (from {ImageStartTokenId}).", name);
        int imageTokens = (ImageSize / PatchSize / 4) * (ImageSize / PatchSize / 4);
        if (imageTokens + 64 > MaxSequenceLength)
            throw new ArgumentException($"{imageTokens} image tokens leave too little of MaxSequenceLength ({MaxSequenceLength}) for the prompt and answer.", nameof(MaxSequenceLength));
        if (MaxGenerationLength <= 0)
            throw new ArgumentException("MaxGenerationLength must be positive.", nameof(MaxGenerationLength));
    }
}
