using AiDotNet.VisionLanguage.RemoteSensing;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Configuration options for SkyEyeGPT.
/// </summary>
public class SkyEyeGPTOptions : RemoteSensingVLMOptions
{
    public SkyEyeGPTOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        SupportedBands = "RGB";
    }

    /// <summary>Gets or sets the number of instruction-tuning samples.</summary>
    public int InstructionSamples { get; set; } = 968000;
}
