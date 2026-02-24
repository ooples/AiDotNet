using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>Options for PromptTTS description-based TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PromptTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class PromptTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PromptTTSOptions(PromptTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        PromptEncoderDim = other.PromptEncoderDim;
        NumPromptLayers = other.NumPromptLayers;
    }

    public PromptTTSOptions()
    {
        NumFlowSteps = 0;
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int PromptEncoderDim { get; set; } = 128;
    public int NumPromptLayers { get; set; } = 3;
}
