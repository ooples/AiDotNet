namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for ProDiff (progressive fast diffusion model for high-quality TTS).</summary>
public class ProDiffOptions : AcousticModelOptions
{
    public ProDiffOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the number of diffusion steps at inference (progressive reduces to 2-4).</summary>
    public int NumDiffusionSteps { get; set; } = 4;

    /// <summary>Gets or sets whether to use knowledge distillation for step reduction.</summary>
    public bool UseProgressiveDistillation { get; set; } = true;
}
