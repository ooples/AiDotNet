using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for NaturalSpeech2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the NaturalSpeech2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class NaturalSpeech2Options : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public NaturalSpeech2Options(NaturalSpeech2Options other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        DiffusionDim = other.DiffusionDim;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public NaturalSpeech2Options()
    {
        NumDiffusionSteps = 100;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the hidden width of the diffusion decoder.</summary>
    public int DiffusionDim { get; set; } = 256;
}
