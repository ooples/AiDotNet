using AiDotNet.VisionLanguage.RemoteSensing;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Configuration options for GeoChat.
/// </summary>
public class GeoChatOptions : RemoteSensingVLMOptions
{
    public GeoChatOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "Vicuna";
        SupportedBands = "RGB";
    }
}
