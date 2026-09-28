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
}