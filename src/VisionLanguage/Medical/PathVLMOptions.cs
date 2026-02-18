using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for PathVLM.
/// </summary>
public class PathVLMOptions : MedicalVLMOptions
{
    public PathVLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        MedicalDomain = "Pathology";
    }

    /// <summary>Gets or sets the pathology image magnification level.</summary>
    public int MagnificationLevel { get; set; } = 20;
}
