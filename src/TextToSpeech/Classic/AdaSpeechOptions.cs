namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AdaSpeech (adaptive TTS with acoustic condition modeling).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AdaSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class AdaSpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AdaSpeechOptions(AdaSpeechOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ConditionDim = other.ConditionDim;
        NumConditionLayers = other.NumConditionLayers;
    }

    public AdaSpeechOptions()
    {
        EncoderDim = 256;
        DecoderDim = 80;
        HiddenDim = 256;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
        NumHeads = 2;
    }

    /// <summary>Gets or sets the acoustic condition embedding dimension.</summary>
    public int ConditionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of conditional layer normalization layers.</summary>
    public int NumConditionLayers { get; set; } = 4;
}
