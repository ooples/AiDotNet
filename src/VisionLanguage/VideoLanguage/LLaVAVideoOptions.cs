namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for LLaVA-Video: synthetic dataset-trained video instruction model.
/// </summary>
public class LLaVAVideoOptions : VideoLanguageOptions
{
    public LLaVAVideoOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "Qwen2";
        MaxFrames = 64;
    }
}
