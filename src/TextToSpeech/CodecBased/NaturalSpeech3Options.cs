using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for NaturalSpeech3 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the NaturalSpeech3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class NaturalSpeech3Options : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public NaturalSpeech3Options(NaturalSpeech3Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        DiffusionDim = other.DiffusionDim;
    }

    public NaturalSpeech3Options()
    {
        NumDiffusionSteps = 100;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    public int DiffusionDim { get; set; } = 256;
}
