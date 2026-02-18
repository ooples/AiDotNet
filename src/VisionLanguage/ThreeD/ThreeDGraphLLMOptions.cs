using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for 3DGraphLLM.
/// </summary>
public class ThreeDGraphLLMOptions : ThreeDVLMOptions
{
    public ThreeDGraphLLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        MaxPoints = 8192;
        PointChannels = 6;
        PointEncoderDim = 768;
    }

    /// <summary>Gets or sets the maximum number of graph nodes.</summary>
    public int MaxGraphNodes { get; set; } = 256;
}
