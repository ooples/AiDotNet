using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for EmotiVoice TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EmotiVoice model. Default values follow the original paper settings.</para>
/// </remarks>
public class EmotiVoiceOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EmotiVoiceOptions(EmotiVoiceOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        EmotionDim = other.EmotionDim;
        NumEmotionLayers = other.NumEmotionLayers;
    }

    public EmotiVoiceOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int EmotionDim { get; set; } = 128;
    public int NumEmotionLayers { get; set; } = 3;
}
