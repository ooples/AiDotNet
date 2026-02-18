using AiDotNet.VisionLanguage.RemoteSensing;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Configuration options for RSGPT.
/// </summary>
public class RSGPTOptions : RemoteSensingVLMOptions
{
    public RSGPTOptions()
    {
        VisionDim = 1408;
        DecoderDim = 4096;
        NumVisionLayers = 39;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Vicuna";
        SupportedBands = "RGB";
    }
}
