using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for Scene-LLM.
/// </summary>
public class SceneLLMOptions : ThreeDVLMOptions
{
    public SceneLLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        MaxPoints = 32768;
        PointChannels = 6;
        PointEncoderDim = 768;
    }

    /// <summary>Gets or sets the voxel grid resolution.</summary>
    public int VoxelResolution { get; set; } = 64;
}
