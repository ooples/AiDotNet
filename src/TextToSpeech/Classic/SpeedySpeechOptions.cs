namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for SpeedySpeech (teacher-student distilled non-autoregressive TTS).</summary>
public class SpeedySpeechOptions : AcousticModelOptions
{
    public SpeedySpeechOptions() { EncoderDim = 128; DecoderDim = 80; HiddenDim = 128; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 2; }

    /// <summary>Gets or sets the convolutional residual block kernel size.</summary>
    public int ResidualKernelSize { get; set; } = 3;
}
