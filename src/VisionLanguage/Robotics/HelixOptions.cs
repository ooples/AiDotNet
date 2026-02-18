using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for Helix.
/// </summary>
public class HelixOptions : VisionLanguageActionOptions
{
    public HelixOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        ActionDimension = 22;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the number of joint DOFs controlled.</summary>
    public int NumJoints { get; set; } = 22;
}
