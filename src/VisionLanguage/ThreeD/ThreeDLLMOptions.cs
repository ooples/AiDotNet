using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for 3D-LLM.
/// </summary>
public class ThreeDLLMOptions : ThreeDVLMOptions
{
    public ThreeDLLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        MaxPoints = 16384;
        PointChannels = 6;
        PointEncoderDim = 768;
    }

    /// <summary>Gets or sets the number of multi-view images for 3D feature extraction.</summary>
    public int NumViews { get; set; } = 8;
}
