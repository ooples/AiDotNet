namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for Video-LLaVA: united visual representation for video understanding.
/// </summary>
public class VideoLLaVAOptions : VideoLanguageOptions
{
    public VideoLLaVAOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        MaxFrames = 8;
    }
}
