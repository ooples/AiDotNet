using AiDotNet.Models.Options;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for GLaMM (Rasheed et al. 2024), the grounding large multimodal model that answers with text and
/// a segmentation mask for every <c>[SEG]</c> it writes.
/// </summary>
/// <remarks>
/// <para>The defaults are the released model's (MBZUAI/GLaMM-GranD-Pretrained, built by the official train.py): CLIP
/// ViT-L/14 at 336 px, Vicuna-7B v1.5 with GLaMM's seven added tokens and LoRA (rank 8, alpha 16, dropout 0.05 on q_proj
/// and v_proj), the region encoder, SAM ViT-H at 1024 px, a 256-wide [SEG] projection, and the loss weights 1 (CE),
/// 2 (BCE) and 0.5 (DICE).</para>
/// <para>GLaMM reads Vicuna's SentencePiece model (<c>tokenizer.model</c>, under the Llama 2 license), which is not
/// shipped: set <see cref="TokenizerPath"/> or <see cref="TokenizerModel"/>.</para>
/// <para><b>For Beginners:</b> These options configure GLaMM's four parts: the image encoder, the language model, the
/// region encoder and the segmenter.</para>
/// </remarks>
public class GLaMMOptions : ModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GLaMMOptions(GLaMMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        TokenizerPath = other.TokenizerPath;
        TokenizerModel = other.TokenizerModel is null ? null : (byte[])other.TokenizerModel.Clone();
        ClipImageSize = other.ClipImageSize;
        ClipPatchSize = other.ClipPatchSize;
        ClipHidden = other.ClipHidden;
        ClipLayers = other.ClipLayers;
        ClipHeads = other.ClipHeads;
        ClipIntermediate = other.ClipIntermediate;
        LlmHidden = other.LlmHidden;
        LlmLayers = other.LlmLayers;
        LlmHeads = other.LlmHeads;
        LlmKeyValueHeads = other.LlmKeyValueHeads;
        LlmIntermediate = other.LlmIntermediate;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        RegionWidth = other.RegionWidth;
        SamImageSize = other.SamImageSize;
        SamPatchSize = other.SamPatchSize;
        SamEncoderDim = other.SamEncoderDim;
        SamEncoderDepth = other.SamEncoderDepth;
        SamEncoderHeads = other.SamEncoderHeads;
        SamWindowSize = other.SamWindowSize;
        SamGlobalAttention = (int[])other.SamGlobalAttention.Clone();
        SamPromptDim = other.SamPromptDim;
        SamMaskInChannels = other.SamMaskInChannels;
        SamDecoderHeads = other.SamDecoderHeads;
        SamDecoderMlpDim = other.SamDecoderMlpDim;
        SamIouHiddenDim = other.SamIouHiddenDim;
        ModelMaxLength = other.ModelMaxLength;
        CeLossWeight = other.CeLossWeight;
        BceLossWeight = other.BceLossWeight;
        DiceLossWeight = other.DiceLossWeight;
    }

    public GLaMMOptions()
    {
    }

    /// <summary>Gets or sets the path of Vicuna's SentencePiece <c>tokenizer.model</c>.</summary>
    public string? TokenizerPath { get; set; }

    /// <summary>Gets or sets the bytes of a SentencePiece BPE model, in place of <see cref="TokenizerPath"/>.</summary>
    public byte[]? TokenizerModel { get; set; }

    /// <summary>Gets or sets CLIP's image side (336).</summary>
    public int ClipImageSize { get; set; } = 336;

    /// <summary>Gets or sets CLIP's patch side (14).</summary>
    public int ClipPatchSize { get; set; } = 14;

    /// <summary>Gets or sets CLIP's width (1024).</summary>
    public int ClipHidden { get; set; } = 1024;

    /// <summary>Gets or sets CLIP's layers (24).</summary>
    public int ClipLayers { get; set; } = 24;

    /// <summary>Gets or sets CLIP's attention heads (16).</summary>
    public int ClipHeads { get; set; } = 16;

    /// <summary>Gets or sets CLIP's feed-forward width (4096).</summary>
    public int ClipIntermediate { get; set; } = 4096;

    /// <summary>Gets or sets the language model's width (4096).</summary>
    public int LlmHidden { get; set; } = 4096;

    /// <summary>Gets or sets the language model's layers (32).</summary>
    public int LlmLayers { get; set; } = 32;

    /// <summary>Gets or sets the language model's attention heads (32).</summary>
    public int LlmHeads { get; set; } = 32;

    /// <summary>Gets or sets the language model's key-value heads (32).</summary>
    public int LlmKeyValueHeads { get; set; } = 32;

    /// <summary>Gets or sets the language model's SwiGLU width (11008).</summary>
    public int LlmIntermediate { get; set; } = 11008;

    /// <summary>Gets or sets the LoRA rank on q_proj and v_proj (8); 0 trains every decoder layer instead.</summary>
    public int LoraRank { get; set; } = 8;

    /// <summary>Gets or sets LoRA alpha (16).</summary>
    public double LoraAlpha { get; set; } = 16;

    /// <summary>Gets or sets LoRA dropout (0.05).</summary>
    public double LoraDropout { get; set; } = 0.05;

    /// <summary>Gets or sets the region encoder's region and position width (1024, fixed in the official code).</summary>
    public int RegionWidth { get; set; } = 1024;

    /// <summary>Gets or sets SAM's image side (1024).</summary>
    public int SamImageSize { get; set; } = 1024;

    /// <summary>Gets or sets SAM's patch side (16).</summary>
    public int SamPatchSize { get; set; } = 16;

    /// <summary>Gets or sets SAM's encoder width (ViT-H: 1280).</summary>
    public int SamEncoderDim { get; set; } = 1280;

    /// <summary>Gets or sets SAM's encoder depth (ViT-H: 32).</summary>
    public int SamEncoderDepth { get; set; } = 32;

    /// <summary>Gets or sets SAM's encoder heads (16).</summary>
    public int SamEncoderHeads { get; set; } = 16;

    /// <summary>Gets or sets SAM's attention window (14).</summary>
    public int SamWindowSize { get; set; } = 14;

    /// <summary>Gets or sets SAM's globally attending blocks (ViT-H: 7, 15, 23, 31).</summary>
    public int[] SamGlobalAttention { get; set; } = [7, 15, 23, 31];

    /// <summary>Gets or sets SAM's prompt and decoder width, also the [SEG] projection's output (256).</summary>
    public int SamPromptDim { get; set; } = 256;

    /// <summary>Gets or sets the channels of SAM's mask embedding (16).</summary>
    public int SamMaskInChannels { get; set; } = 16;

    /// <summary>Gets or sets SAM's mask decoder heads (8).</summary>
    public int SamDecoderHeads { get; set; } = 8;

    /// <summary>Gets or sets SAM's mask decoder feed-forward width (2048).</summary>
    public int SamDecoderMlpDim { get; set; } = 2048;

    /// <summary>Gets or sets SAM's IoU head width (256).</summary>
    public int SamIouHiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the tokenizer's maximum length (1536); training keeps the first 1536 − 575 ids, as the
    /// official collate does.</summary>
    public int ModelMaxLength { get; set; } = 1536;

    /// <summary>Gets or sets the language-model loss weight (1).</summary>
    public double CeLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the mask cross-entropy weight (2).</summary>
    public double BceLossWeight { get; set; } = 2.0;

    /// <summary>Gets or sets the mask DICE weight (0.5).</summary>
    public double DiceLossWeight { get; set; } = 0.5;
}
