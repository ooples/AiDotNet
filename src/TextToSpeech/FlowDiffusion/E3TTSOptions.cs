using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for E3TTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the E3TTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class E3TTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public E3TTSOptions(E3TTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        DiffusionDim = other.DiffusionDim;
    }

    public E3TTSOptions()
    {
        NumDiffusionSteps = 50;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    public int DiffusionDim { get; set; } = 256;
}
