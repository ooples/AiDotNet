namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoLLaMA 3: frontier multimodal for image and video.
/// </summary>
public class VideoLLaMA3Options : VideoLanguageOptions
{
    public VideoLLaMA3Options()
    {
        VisionDim = 1152;
        DecoderDim = 4096;
        NumVisionLayers = 27;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 128256;
        LanguageModelName = "LLaMA-3";
        MaxFrames = 128;
    }
}
