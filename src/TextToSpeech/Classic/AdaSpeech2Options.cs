namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AdaSpeech 2 (adaptive TTS with untranscribed speech data).</summary>
/// <remarks>
/// <para>AdaSpeech 2's TTS pipeline is AdaSpeech's (Yan et al. 2021, §2.1, §3.1: hidden 256, 2 heads, FFN filter 1024
/// and kernel 9, 80-bin mel, 16 kHz audio with a 12.5 ms hop), so these options extend <see cref="AdaSpeechOptions"/>
/// with the mel-spectrogram encoder.</para>
/// <para><b>For Beginners:</b> These options configure the AdaSpeech2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class AdaSpeech2Options : AdaSpeechOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AdaSpeech2Options(AdaSpeech2Options other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumMelEncoderLayers = other.NumMelEncoderLayers;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public AdaSpeech2Options()
    {
    }

    /// <summary>Gets or sets the feed-forward Transformer blocks of the mel-spectrogram encoder (4, symmetric with the
    /// phoneme encoder, §2.2).</summary>
    public int NumMelEncoderLayers { get; set; } = 4;
}
