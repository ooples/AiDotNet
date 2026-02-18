using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for RT-2.
/// </summary>
public class RT2Options : VisionLanguageActionOptions
{
    public RT2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "PaLI-X";
        ActionDimension = 7;
        PredictionHorizon = 6;
    }
}
