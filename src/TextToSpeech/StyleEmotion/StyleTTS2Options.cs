using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for StyleTTS2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the StyleTTS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class StyleTTS2Options : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StyleTTS2Options(StyleTTS2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        StyleDim = other.StyleDim;
        NumStyleDiffusionSteps = other.NumStyleDiffusionSteps;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public StyleTTS2Options()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the width of the style vector that conditions synthesis.</summary>
    public int StyleDim { get; set; } = 128;
    /// <summary>Gets or sets the number of denoising steps used to sample the style vector.</summary>
    public int NumStyleDiffusionSteps { get; set; } = 5;
}
