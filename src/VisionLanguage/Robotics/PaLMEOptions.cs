using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for PaLM-E.
/// </summary>
public class PaLMEOptions : VisionLanguageActionOptions
{
    public PaLMEOptions()
    {
        VisionDim = 1408;
        DecoderDim = 8192;
        NumVisionLayers = 48;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 224;
        VocabSize = 256000;
        LanguageModelName = "PaLM";
        ActionDimension = 7;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 562;
}
