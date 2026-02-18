using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for pi-zero.
/// </summary>
public class PiZeroOptions : VisionLanguageActionOptions
{
    public PiZeroOptions()
    {
        VisionDim = 1152;
        DecoderDim = 2048;
        NumVisionLayers = 27;
        NumDecoderLayers = 18;
        NumHeads = 16;
        ImageSize = 224;
        VocabSize = 256000;
        LanguageModelName = "PaliGemma";
        ActionDimension = 7;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the number of flow matching denoising steps.</summary>
    public int NumFlowSteps { get; set; } = 10;
}
