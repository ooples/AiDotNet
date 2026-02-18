using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for Octo.
/// </summary>
public class OctoOptions : VisionLanguageActionOptions
{
    public OctoOptions()
    {
        VisionDim = 384;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Transformer";
        ActionDimension = 7;
        PredictionHorizon = 4;
        ObservationHistory = 2;
    }

    /// <summary>Gets or sets the total parameter count in millions.</summary>
    public int TotalParameters { get; set; } = 93;
}
