namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveGlow (flow-based vocoder combining Glow and WaveNet).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WaveGlow model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveGlowOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveGlowOptions(WaveGlowOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumFlows = other.NumFlows;
        NumWaveNetLayers = other.NumWaveNetLayers;
        EarlyOutputChannels = other.EarlyOutputChannels;
    }
 public WaveGlowOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; } public int NumFlows { get; set; } = 12; public int NumWaveNetLayers { get; set; } = 8; public int EarlyOutputChannels { get; set; } = 2; }
