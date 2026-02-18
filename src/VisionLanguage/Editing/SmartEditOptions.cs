namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Configuration options for SmartEdit: enhanced instruction understanding for complex image editing.
/// </summary>
public class SmartEditOptions : EditingVLMOptions
{
    public SmartEditOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to enable complex reasoning for editing instructions.</summary>
    public bool EnableComplexReasoning { get; set; } = true;
}
