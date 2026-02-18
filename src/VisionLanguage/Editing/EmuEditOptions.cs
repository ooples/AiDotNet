namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Configuration options for Emu Edit: precise image editing via recognition and generation tasks.
/// </summary>
public class EmuEditOptions : EditingVLMOptions
{
    public EmuEditOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use recognition-guided precise editing.</summary>
    public bool EnablePreciseEditing { get; set; } = true;
}
