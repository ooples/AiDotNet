namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveNet (autoregressive dilated causal convolution vocoder).</summary>
public class WaveNetOptions : VocoderOptions
{
    public WaveNetOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; }
    /// <summary>Gets or sets the number of dilated causal convolution layers.</summary>
    public int NumDilatedLayers { get; set; } = 30;
    /// <summary>Gets or sets the residual channel count.</summary>
    public int ResidualChannels { get; set; } = 64;
    /// <summary>Gets or sets the skip channel count.</summary>
    public int SkipChannels { get; set; } = 256;
    /// <summary>Gets or sets the number of mu-law quantization levels.</summary>
    public int MuLawLevels { get; set; } = 256;
}
