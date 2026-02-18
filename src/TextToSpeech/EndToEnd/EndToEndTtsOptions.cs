namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Base options for end-to-end TTS models that generate waveforms directly from text.</summary>
public class EndToEndTtsOptions : TtsModelOptions
{
    public EndToEndTtsOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; HiddenDim = 192; }
    /// <summary>Gets or sets the number of flow steps in the posterior encoder.</summary>
    public int NumFlowSteps { get; set; } = 4;
    /// <summary>Gets or sets the number of diffusion/denoising steps (if applicable).</summary>
    public int NumDiffusionSteps { get; set; } = 50;
    /// <summary>Gets or sets the inter-channel dimension for the normalizing flow.</summary>
    public int InterChannels { get; set; } = 192;
    /// <summary>Gets or sets the filter channels for the text encoder.</summary>
    public int FilterChannels { get; set; } = 768;
    /// <summary>Gets or sets the encoder hidden dimension (defaults to HiddenDim).</summary>
    public int EncoderDim { get; set; } = 192;
    /// <summary>Gets or sets the decoder hidden dimension (defaults to HiddenDim).</summary>
    public int DecoderDim { get; set; } = 192;
}
