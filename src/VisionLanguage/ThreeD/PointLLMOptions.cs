using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for PointLLM.
/// </summary>
public class PointLLMOptions : ThreeDVLMOptions
{
    public PointLLMOptions()
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
}
