using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for StyleTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the StyleTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class StyleTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StyleTTSOptions(StyleTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        StyleDim = other.StyleDim;
        NumStyleDiffusionSteps = other.NumStyleDiffusionSteps;
    }

    public StyleTTSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int StyleDim { get; set; } = 128;
    public int NumStyleDiffusionSteps { get; set; } = 5;
}
