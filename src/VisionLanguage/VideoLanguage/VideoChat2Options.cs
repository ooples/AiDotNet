namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoChat2: progressive video training with diverse data.
/// </summary>
public class VideoChat2Options : VideoLanguageOptions
{
    public VideoChat2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Mistral";
        MaxFrames = 16;
    }
}
