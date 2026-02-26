namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for SpeedySpeech (teacher-student distilled non-autoregressive TTS).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpeedySpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeedySpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SpeedySpeechOptions(SpeedySpeechOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ResidualKernelSize = other.ResidualKernelSize;
    }

    public SpeedySpeechOptions() { EncoderDim = 128; DecoderDim = 80; HiddenDim = 128; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 2; }

    /// <summary>Gets or sets the convolutional residual block kernel size.</summary>
    public int ResidualKernelSize { get; set; } = 3;
}
