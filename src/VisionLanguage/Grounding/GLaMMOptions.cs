namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for GLaMM: pixel-level grounded LMM generating text and segmentation masks.
/// </summary>
public class GLaMMOptions : GroundingVLMOptions
{
    public GLaMMOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        MaxDetections = 100;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to produce pixel-level grounding masks.</summary>
    public bool EnablePixelGrounding { get; set; } = true;

    /// <summary>Gets or sets the segmentation mask feature dimension.</summary>
    public int MaskDim { get; set; } = 256;
}
