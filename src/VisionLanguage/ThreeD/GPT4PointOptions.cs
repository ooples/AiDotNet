using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for GPT4Point.
/// </summary>
public class GPT4PointOptions : ThreeDVLMOptions
{
    public GPT4PointOptions()
    {
        VisionDim = 512;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        MaxPoints = 8192;
        PointChannels = 6;
        PointEncoderDim = 512;
    }

    /// <summary>Gets or sets whether the model supports point cloud generation.</summary>
    public bool SupportsPointGeneration { get; set; } = true;
}
