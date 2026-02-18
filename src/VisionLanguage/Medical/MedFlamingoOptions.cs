using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for Med-Flamingo.
/// </summary>
public class MedFlamingoOptions : MedicalVLMOptions
{
    public MedFlamingoOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "MPT";
        MedicalDomain = "General";
    }

    /// <summary>Gets or sets the number of few-shot examples supported.</summary>
    public int MaxFewShotExamples { get; set; } = 8;
}
