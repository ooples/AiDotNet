namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Show-o: single transformer for unified understanding and generation.
/// </summary>
public class ShowOOptions : UnifiedVisionOptions
{
    public ShowOOptions()
    {
        VisionDim = 1024;
        DecoderDim = 2048;
        NumVisionLayers = 24;
        NumDecoderLayers = 24;
        NumHeads = 16;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Phi-1.5";
        NumVisualTokens = 8192;
    }
}
