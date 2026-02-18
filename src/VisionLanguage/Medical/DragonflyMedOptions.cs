using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for Dragonfly-Med.
/// </summary>
public class DragonflyMedOptions : MedicalVLMOptions
{
    public DragonflyMedOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MedicalDomain = "General";
    }
}
