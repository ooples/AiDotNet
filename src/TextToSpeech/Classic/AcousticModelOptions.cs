namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Base configuration options for classic acoustic TTS models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AcousticModel model. Default values follow the original paper settings.</para>
/// </remarks>
public class AcousticModelOptions : TtsModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public AcousticModelOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AcousticModelOptions(AcousticModelOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
        OutputsPerStep = other.OutputsPerStep;
        UsePostnet = other.UsePostnet;
        PostnetDim = other.PostnetDim;
        PostnetLayers = other.PostnetLayers;
        TextSymbols = other.TextSymbols is null ? null : (string[])other.TextSymbols.Clone();
    }

    /// <summary>Gets or sets the encoder embedding dimension.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron (Wang et al. 2017) sizes its parts with TacotronOptions' own fields (EmbeddingDim, CbhgChannels); the inherited encoder width has no part to size.", ModelType = typeof(Tacotron<>))]
    public virtual int EncoderDim { get; set; } = 256;

    /// <summary>Ordered symbol inventory used to train the text embedding. Null selects this model's English inventory.</summary>
    public string[]? TextSymbols { get; set; }

    /// <summary>Gets or sets the decoder output dimension (mel channels).</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron (Wang et al. 2017) sizes its parts with TacotronOptions' own fields (DecoderRnnDim, AttentionDim); the inherited decoder width has no part to size.", ModelType = typeof(Tacotron<>))]
    public virtual int DecoderDim { get; set; } = 80;

    /// <summary>Gets or sets the number of mel frames generated per decoder step (reduction factor).</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("PromptTTS's acoustic model is FastSpeech 2's (Guo et al. 2023): non-autoregressive, it emits every frame at once, so there is no reduction factor.", ModelType = typeof(DescriptionBased.PromptTTS<>))]
    public int OutputsPerStep { get; set; } = 1;

    /// <summary>Gets or sets whether to use a postnet for mel refinement.</summary>
    public bool UsePostnet { get; set; } = true;

    /// <summary>Gets or sets the postnet embedding dimension.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("PromptTTS's acoustic model is FastSpeech 2's (Guo et al. 2023), which has no post-net (FastSpeech2Options sets UsePostnet = false).", ModelType = typeof(DescriptionBased.PromptTTS<>))]
    public int PostnetDim { get; set; } = 512;

    /// <summary>Gets or sets the number of postnet convolution layers.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("PromptTTS's acoustic model is FastSpeech 2's (Guo et al. 2023), which has no post-net (FastSpeech2Options sets UsePostnet = false).", ModelType = typeof(DescriptionBased.PromptTTS<>))]
    public int PostnetLayers { get; set; } = 5;
}
