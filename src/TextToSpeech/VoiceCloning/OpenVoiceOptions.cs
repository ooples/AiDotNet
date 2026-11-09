using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for OpenVoice TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OpenVoice model. Default values follow the original paper settings.</para>
/// </remarks>
public class OpenVoiceOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OpenVoiceOptions(OpenVoiceOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        NumToneColorLayers = other.NumToneColorLayers;
        Training = new OpenVoiceTrainingOptions(other.Training);
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public OpenVoiceOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the width of the speaker embedding.</summary>
    public int SpeakerEmbeddingDim { get; set; } = 256;
    /// <summary>Gets or sets the number of layers in the tone-color converter.</summary>
    public int NumToneColorLayers { get; set; } = 3;
    /// <summary>Number of layers in the auxiliary IPA training encoder. Shares the setting with <see cref="Training"/>.</summary>
    public override int NumEncoderLayers
    {
        get => Training?.IpaLayers ?? base.NumEncoderLayers;
        set
        {
            base.NumEncoderLayers = value;
            if (Training is not null) Training.IpaLayers = value;
        }
    }

    /// <summary>Attention heads in the auxiliary IPA training encoder. Shares the setting with <see cref="Training"/>.</summary>
    public override int NumHeads
    {
        get => Training?.IpaHeads ?? base.NumHeads;
        set
        {
            base.NumHeads = value;
            if (Training is not null) Training.IpaHeads = value;
        }
    }

    /// <summary>Dropout probability in the auxiliary IPA training encoder. Shares the setting with <see cref="Training"/>.</summary>
    public override double DropoutRate
    {
        get => Training?.IpaDropout ?? base.DropoutRate;
        set
        {
            base.DropoutRate = value;
            if (Training is not null) Training.IpaDropout = value;
        }
    }

    /// <summary>Number of mel bands used by the waveform reconstruction objective. Shares the setting with <see cref="Training"/>.</summary>
    public override int MelChannels
    {
        get => Training?.MelChannels ?? base.MelChannels;
        set
        {
            base.MelChannels = value;
            if (Training is not null) Training.MelChannels = value;
        }
    }
    /// <summary>Auxiliary IPA prior, waveform reconstruction and adversarial training configuration.</summary>
    public OpenVoiceTrainingOptions Training { get; set; } = new();
}
