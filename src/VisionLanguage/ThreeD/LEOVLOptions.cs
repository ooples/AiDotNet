using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for LEO-VL.
/// </summary>
public class LEOVLOptions : ThreeDVLMOptions
{
    public LEOVLOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MaxPoints = 8192;
        PointChannels = 9;
        PointEncoderDim = 1024;
    }

    /// <summary>Gets or sets the number of RGB-D views for scene reconstruction.</summary>
    public int NumViews { get; set; } = 12;
}
