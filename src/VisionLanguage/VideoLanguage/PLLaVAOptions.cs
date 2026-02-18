namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for PLLaVA: parameter-free pooling extension from images to video.
/// </summary>
public class PLLaVAOptions : VideoLanguageOptions
{
    public PLLaVAOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "Vicuna";
        MaxFrames = 16;
    }

    /// <summary>Gets or sets whether to use parameter-free pooling for frame compression.</summary>
    public bool EnableParameterFreePooling { get; set; } = true;
}
