namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Configuration options for MGIE: MLLM-guided image editing with LLaVA-based instruction understanding.
/// </summary>
public class MGIEOptions : EditingVLMOptions
{
    public MGIEOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use MLLM for expressive instruction understanding.</summary>
    public bool EnableExpressiveInstructions { get; set; } = true;
}
