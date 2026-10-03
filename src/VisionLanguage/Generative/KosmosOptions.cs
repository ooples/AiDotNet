using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// Hyperparameters shared by KOSMOS-1 and KOSMOS-2:
/// <list type="bullet">
/// <item>A CLIP ViT-L/14 image encoder: 224 px, 1024 wide, 24 layers, 16 heads, patch 14.</item>
/// <item>A 64-token image resampler.</item>
/// <item>A MAGNETO causal decoder: 2048 wide, 24 layers, 32 heads, FFN 8192, 2048 positions.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The defaults reproduce the published models. The token ids of the image
/// markers must match the tokenizer used to train the model.</para>
/// </remarks>
public abstract class KosmosOptions : GenerativeVLMOptions
{
    /// <summary>Initializes the options with the shared KOSMOS values.</summary>
    protected KosmosOptions()
    {
        ArchitectureType = GenerativeArchitectureType.CausalMultimodal;
        ImageSize = 224;
        VisionDim = 1024;
        DecoderDim = 2048;
        NumVisionLayers = 24;
        NumDecoderLayers = 24;
        NumHeads = 32;
        MaxSequenceLength = 2048;
        VisionHeads = 16;
        PatchSize = 14;
        NumImageTokens = 64;
        DecoderFeedForwardDim = 8192;
        BosTokenId = 0;
        EosTokenId = 2;
        ImageStartTokenId = -1;
        ImageEndTokenId = -1;
    }

    /// <summary>Copies another instance's values.</summary>
    protected KosmosOptions(KosmosOptions other)
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
        VisionHeads = other.VisionHeads;
        PatchSize = other.PatchSize;
        NumImageTokens = other.NumImageTokens;
        DecoderFeedForwardDim = other.DecoderFeedForwardDim;
        BosTokenId = other.BosTokenId;
        EosTokenId = other.EosTokenId;
        ImageStartTokenId = other.ImageStartTokenId;
        ImageEndTokenId = other.ImageEndTokenId;
    }

    /// <summary>Gets or sets the vision encoder's attention heads. CLIP ViT-L/14: 16.</summary>
    public int VisionHeads { get; set; }

    /// <summary>Gets or sets the vision encoder's patch size. CLIP ViT-L/14: 14.</summary>
    public int PatchSize { get; set; }

    /// <summary>Gets or sets the number of image embeddings placed in the sequence (latent queries). Paper: 64.</summary>
    public int NumImageTokens { get; set; }

    /// <summary>Gets or sets the decoder feed-forward width. Paper: 8192.</summary>
    public int DecoderFeedForwardDim { get; set; }

    /// <summary>Gets or sets the begin-of-sequence token id. Reference: 0.</summary>
    public int BosTokenId { get; set; }

    /// <summary>Gets or sets the end-of-sequence token id. Reference: 2.</summary>
    public int EosTokenId { get; set; }

    /// <summary>
    /// Gets or sets the <c>&lt;image&gt;</c> token id. -1 (default) means VocabSize - 2; set it to the
    /// tokenizer's id when loading trained weights.
    /// </summary>
    public int ImageStartTokenId { get; set; }

    /// <summary>
    /// Gets or sets the <c>&lt;/image&gt;</c> token id. -1 (default) means VocabSize - 1; set it to the
    /// tokenizer's id when loading trained weights.
    /// </summary>
    public int ImageEndTokenId { get; set; }

    /// <summary>The effective <c>&lt;image&gt;</c> id.</summary>
    internal int ResolvedImageStartTokenId => ImageStartTokenId >= 0 ? ImageStartTokenId : VocabSize - 2;

    /// <summary>The effective <c>&lt;/image&gt;</c> id.</summary>
    internal int ResolvedImageEndTokenId => ImageEndTokenId >= 0 ? ImageEndTokenId : VocabSize - 1;
}
