namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Glow-TTS (flow-based non-autoregressive TTS with monotonic alignment search).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GlowTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class GlowTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GlowTTSOptions(GlowTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumFlowLayers = other.NumFlowLayers;
        Temperature = other.Temperature;
    }

    public GlowTTSOptions() { EncoderDim = 192; DecoderDim = 80; HiddenDim = 192; NumEncoderLayers = 6; NumDecoderLayers = 12; NumHeads = 2; }

    /// <summary>Gets or sets the number of flow coupling layers.</summary>
    public int NumFlowLayers { get; set; } = 12;

    /// <summary>Gets or sets the temperature for sampling during inference.</summary>
    public double Temperature { get; set; } = 0.333;
}
