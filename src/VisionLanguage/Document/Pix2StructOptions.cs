namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Pix2Struct: screenshot parsing pre-training for visual language understanding.
/// </summary>
public class Pix2StructOptions : DocumentVLMOptions
{
    public Pix2StructOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 1024;
    }

    /// <summary>Gets or sets the maximum number of image patches.</summary>
    public int MaxPatchesPerImage { get; set; } = 2048;

    /// <summary>Gets or sets whether to use variable-resolution patching.</summary>
    public bool EnableVariableResolution { get; set; } = true;
}
