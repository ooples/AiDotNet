namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Glow-TTS (flow-based non-autoregressive TTS with monotonic alignment search).</summary>
public class GlowTTSOptions : AcousticModelOptions
{
    public GlowTTSOptions() { EncoderDim = 192; DecoderDim = 80; HiddenDim = 192; NumEncoderLayers = 6; NumDecoderLayers = 12; NumHeads = 2; }

    /// <summary>Gets or sets the number of flow coupling layers.</summary>
    public int NumFlowLayers { get; set; } = 12;

    /// <summary>Gets or sets the temperature for sampling during inference.</summary>
    public double Temperature { get; set; } = 0.333;
}
