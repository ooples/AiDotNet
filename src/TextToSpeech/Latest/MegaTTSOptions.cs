using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for MegaTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MegaTTS model. Default values follow the original paper settings.</para>
/// <para>
/// Mega-TTS (Jiang et al. 2023, arXiv:2306.03509) decomposes speech into four attributes and
/// models each with the inductive bias that suits it: <b>content</b> from the phoneme sequence,
/// <b>timbre</b> as a single time-invariant global vector, <b>prosody</b> as a tightly
/// bottlenecked time-varying latent, and <b>phase</b> left entirely to the vocoder. The options
/// below expose every dimension of that decomposition; the defaults are the paper's.
/// </para>
/// </remarks>
public class MegaTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MegaTTSOptions(MegaTTSOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ProsodyDim = other.ProsodyDim;
        ProsodyCodebookSize = other.ProsodyCodebookSize;
        NumProsodyLayers = other.NumProsodyLayers;
        ProsodyMelBands = other.ProsodyMelBands;
        TimbreDim = other.TimbreDim;
        NumTimbreLayers = other.NumTimbreLayers;
        PLLMDim = other.PLLMDim;
        NumPLLMLayers = other.NumPLLMLayers;
        NumPLLMHeads = other.NumPLLMHeads;
    }

    public MegaTTSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    /// <summary>
    /// Gets or sets the width of the vector-quantized prosody latent (paper: 20).
    /// </summary>
    /// <remarks>
    /// Deliberately tiny. Mega-TTS §3.2 relies on a hard information bottleneck here: the prosody
    /// code must be too narrow to smuggle timbre or content through, which is what forces those
    /// attributes into their own branches. Widening this is the single fastest way to break the
    /// disentanglement the model is built around.
    /// </remarks>
    public int ProsodyDim { get; set; } = 20;

    /// <summary>Gets or sets the number of entries in the prosody VQ codebook (paper: 1024).</summary>
    public int ProsodyCodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the number of prosody-encoder blocks (paper: 2).</summary>
    public int NumProsodyLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets how many of the lowest mel bands the prosody encoder sees (paper: 20 of 80).
    /// </summary>
    /// <remarks>
    /// The second inductive bias of Mega-TTS §3.2: prosody (pitch/energy/rhythm) lives in the low
    /// frequencies, while timbre is carried by the higher formant structure. Restricting the
    /// prosody encoder's input band is what stops it from learning speaker identity.
    /// </remarks>
    public int ProsodyMelBands { get; set; } = 20;

    /// <summary>Gets or sets the width of the global timbre vector (paper: 192).</summary>
    public int TimbreDim { get; set; } = 192;

    /// <summary>Gets or sets the number of timbre-encoder blocks (paper: 2).</summary>
    public int NumTimbreLayers { get; set; } = 2;

    /// <summary>Gets or sets the width of the prosody latent language model (paper: 512).</summary>
    public int PLLMDim { get; set; } = 512;

    /// <summary>Gets or sets the depth of the prosody latent language model (paper: 4).</summary>
    public int NumPLLMLayers { get; set; } = 4;

    /// <summary>Gets or sets the attention heads in the prosody latent language model (paper: 8).</summary>
    public int NumPLLMHeads { get; set; } = 8;
}
