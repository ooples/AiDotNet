using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for StyleTTSZS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the StyleTTSZS model. Default values follow the original paper settings.</para>
/// </remarks>
public class StyleTTSZSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StyleTTSZSOptions(StyleTTSZSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        StyleDim = other.StyleDim;
        NumStyleLayers = other.NumStyleLayers;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public StyleTTSZSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the width of the style vector that conditions synthesis.</summary>
    public int StyleDim { get; set; } = 256;
    /// <summary>Gets or sets the number of layers in the style encoder.</summary>
    public int NumStyleLayers { get; set; } = 4;
}
