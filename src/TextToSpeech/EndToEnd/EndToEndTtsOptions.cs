namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Base options for end-to-end TTS models that generate waveforms directly from text.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EndToEndTts model. Default values follow the original paper settings.</para>
/// </remarks>
public class EndToEndTtsOptions : TtsModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EndToEndTtsOptions(EndToEndTtsOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumFlowSteps = other.NumFlowSteps;
        NumDiffusionSteps = other.NumDiffusionSteps;
        InterChannels = other.InterChannels;
        FilterChannels = other.FilterChannels;
        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public EndToEndTtsOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        HiddenDim = 192;
    }

    /// <summary>Gets or sets the number of flow steps in the posterior encoder.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro's duration predictor and ISTFTNet decoder contain no normalizing flow.", ModelType = typeof(Kokoro<>))]
    public int NumFlowSteps { get; set; } = 4;

    /// <summary>Gets or sets the number of diffusion/denoising steps (if applicable).</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro selects released voice styles directly and contains no diffusion sampler.", ModelType = typeof(Kokoro<>))]
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>Gets or sets the inter-channel dimension for the normalizing flow.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro contains no flow latent representation whose channels this inherited option configures.", ModelType = typeof(Kokoro<>))]
    public int InterChannels { get; set; } = 192;

    /// <summary>Gets or sets the filter channels for the text encoder.</summary>
    public int FilterChannels { get; set; } = 768;

    /// <summary>Gets or sets the shared encoder/decoder width through <see cref="TtsModelOptions.HiddenDim"/>.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("This aliases HiddenDim; Kokoro's released decoder interfaces require width 512, enforced by the native constructor.", ModelType = typeof(Kokoro<>))]
    public int EncoderDim { get => HiddenDim; set => HiddenDim = value; }

    /// <summary>Gets or sets the shared encoder/decoder width through <see cref="TtsModelOptions.HiddenDim"/>.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("This aliases HiddenDim; Kokoro's released decoder interfaces require width 512, enforced by the native constructor.", ModelType = typeof(Kokoro<>))]
    public virtual int DecoderDim { get => HiddenDim; set => HiddenDim = value; }
}
