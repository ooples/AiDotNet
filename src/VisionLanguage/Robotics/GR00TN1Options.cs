using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for GR00T N1.
/// </summary>
public class GR00TN1Options : VisionLanguageActionOptions
{
    public GR00TN1Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Eagle";
        ActionDimension = 52;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the number of humanoid joints controlled.</summary>
    public int NumJoints { get; set; } = 52;
}
