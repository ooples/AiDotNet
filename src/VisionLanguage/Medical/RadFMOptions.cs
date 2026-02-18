using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for RadFM.
/// </summary>
public class RadFMOptions : MedicalVLMOptions
{
    public RadFMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        MedicalDomain = "Radiology";
    }

    /// <summary>Gets or sets whether the model supports 3D volumetric input.</summary>
    public bool Supports3DInput { get; set; } = true;
}
