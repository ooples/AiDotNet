namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Grad-TTS (diffusion-based acoustic model with score matching).</summary>
public class GradTTSOptions : AcousticModelOptions
{
    public GradTTSOptions() { EncoderDim = 192; DecoderDim = 80; HiddenDim = 192; NumEncoderLayers = 6; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the number of diffusion steps at inference.</summary>
    public int NumDiffusionSteps { get; set; } = 10;

    /// <summary>Gets or sets the noise schedule beta start.</summary>
    public double BetaStart { get; set; } = 0.05;

    /// <summary>Gets or sets the noise schedule beta end.</summary>
    public double BetaEnd { get; set; } = 20.0;
}
