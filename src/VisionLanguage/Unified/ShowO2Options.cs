namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Show-o2: improved native unified multimodal model.
/// </summary>
public class ShowO2Options : UnifiedVisionOptions
{
    public ShowO2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
        LanguageModelName = "Qwen2";
        NumVisualTokens = 16384;
    }
}
