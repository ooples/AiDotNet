using AiDotNet.ComputerVision.OCR;

namespace AiDotNet.ComputerVision.OCR.Recognition;

/// <summary>
/// TrOCR's architecture hyperparameters. The defaults are TrOCR-Base (Li et al. 2023, Table 1): a BEiT-Base
/// image encoder and a RoBERTa-Large-sized text decoder.
/// </summary>
/// <remarks>
/// The encoder and decoder have different widths in every published TrOCR size, so each is configured on
/// its own. When they differ, the encoder output is projected to the decoder width before
/// cross-attention, as HuggingFace's VisionEncoderDecoderModel does with <c>enc_to_dec_proj</c>.
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class TrOCROptions<T> : OCROptions<T>
{
    /// <summary>Creates TrOCR-Base options.</summary>
    public TrOCROptions()
    {
        EncoderHiddenDim = 768;
        EncoderHeads = 12;
        EncoderLayers = 12;
        DecoderHiddenDim = 1024;
        DecoderHeads = 16;
        DecoderLayers = 12;
        PatchSize = 16;
    }

    /// <summary>Image-encoder width (BEiT-Base: 768).</summary>
    public int EncoderHiddenDim { get; set; }

    /// <summary>Image-encoder attention heads (BEiT-Base: 12).</summary>
    public int EncoderHeads { get; set; }

    /// <summary>Image-encoder layers (BEiT-Base: 12).</summary>
    public int EncoderLayers { get; set; }

    /// <summary>Text-decoder width (TrOCR-Base: 1024).</summary>
    public int DecoderHiddenDim { get; set; }

    /// <summary>Text-decoder attention heads (TrOCR-Base: 16).</summary>
    public int DecoderHeads { get; set; }

    /// <summary>Text-decoder layers (TrOCR-Base: 12).</summary>
    public int DecoderLayers { get; set; }

    /// <summary>Square image-patch side in pixels (16).</summary>
    public int PatchSize { get; set; }

    /// <summary>
    /// Throws when a size cannot build a working TrOCR: a non-positive width, head count, patch size or layer
    /// count, or a width the head count does not divide (attention would split it into mismatched heads).
    /// </summary>
    /// <exception cref="ArgumentException">A size is invalid; the message names the option.</exception>
    public void Validate()
    {
        RequirePositive(EncoderHiddenDim, nameof(EncoderHiddenDim));
        RequirePositive(EncoderHeads, nameof(EncoderHeads));
        RequirePositive(DecoderHiddenDim, nameof(DecoderHiddenDim));
        RequirePositive(DecoderHeads, nameof(DecoderHeads));
        RequirePositive(PatchSize, nameof(PatchSize));
        if (EncoderLayers < 0)
            throw new ArgumentException($"TrOCROptions.EncoderLayers is {EncoderLayers}; it cannot be negative.", "options");
        if (DecoderLayers < 0)
            throw new ArgumentException($"TrOCROptions.DecoderLayers is {DecoderLayers}; it cannot be negative.", "options");
        RequireDivisible(EncoderHiddenDim, EncoderHeads, nameof(EncoderHiddenDim), nameof(EncoderHeads));
        RequireDivisible(DecoderHiddenDim, DecoderHeads, nameof(DecoderHiddenDim), nameof(DecoderHeads));
    }

    private static void RequirePositive(int value, string name)
    {
        if (value <= 0)
            throw new ArgumentException($"TrOCROptions.{name} is {value}; it must be greater than zero.", "options");
    }

    private static void RequireDivisible(int width, int heads, string widthName, string headsName)
    {
        if (width % heads != 0)
            throw new ArgumentException(
                $"TrOCROptions.{widthName} ({width}) must be divisible by {headsName} ({heads}).", "options");
    }
}