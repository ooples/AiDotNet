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
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        DiffusionDim = other.DiffusionDim;
        WarmupSteps = other.WarmupSteps;
    }

    public NaturalSpeech3Options()
    {
        NumDiffusionSteps = 100;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    public int DiffusionDim { get; set; } = 256;

    /// <summary>
    /// Gets or sets the warmup length of the paper's AdamW recipe (inverse-square-root schedule).
    /// </summary>
    /// <value>Defaults to 5000, the paper's warmup.</value>
    /// <remarks>
    /// The recipe ramps the learning rate up to 1e-4 over this many steps, so a run far shorter than the
    /// warmup barely trains. Lower it for short fine-tuning runs; leave it at the paper value to reproduce
    /// the paper.
    /// </remarks>
    public int WarmupSteps { get; set; } = 5000;
}
