namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoLLaMA 2: spatial-temporal convolution for video tokens.
/// </summary>
public class VideoLLaMA2Options : VideoLanguageOptions
{
    public VideoLLaMA2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "Mistral";
        MaxFrames = 16;
    }

    /// <summary>Gets or sets whether to use spatial-temporal convolution for video token compression.</summary>
    public bool EnableSpatialTemporalConv { get; set; } = true;
}
