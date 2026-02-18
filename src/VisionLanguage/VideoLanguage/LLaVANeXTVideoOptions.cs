namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for LLaVA-NeXT-Video: average pooling for frame token reduction.
/// </summary>
public class LLaVANeXTVideoOptions : VideoLanguageOptions
{
    public LLaVANeXTVideoOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MaxFrames = 32;
    }
}
