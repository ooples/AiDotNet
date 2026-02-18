namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Chameleon: early fusion with discrete tokens for all modalities.
/// </summary>
public class ChameleonOptions : UnifiedVisionOptions
{
    public ChameleonOptions()
    {
        VisionDim = 4096;
        DecoderDim = 4096;
        NumVisionLayers = 0;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 65536;
        LanguageModelName = "Chameleon";
        NumVisualTokens = 8192;
    }
}
