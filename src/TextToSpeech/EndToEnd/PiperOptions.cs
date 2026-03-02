namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for Piper (lightweight VITS-based TTS optimized for edge/embedded deployment).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Piper model. Default values follow the original paper settings.</para>
/// </remarks>
public class PiperOptions : EndToEndTtsOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PiperOptions(PiperOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LengthScale = other.LengthScale;
        NoiseScale = other.NoiseScale;
        NoiseScaleW = other.NoiseScaleW;
    }
 public PiperOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; NumEncoderLayers = 4; } public double LengthScale { get; set; } = 1.0; public double NoiseScale { get; set; } = 0.667; public double NoiseScaleW { get; set; } = 0.8; }
