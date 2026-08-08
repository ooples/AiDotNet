using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// Options for the RWKV streaming transducer, An and Zhang, "Exploring RWKV for Memory Efficient
/// and Low Latency Streaming ASR" (arXiv:2309.14758).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the RWKV streaming recognizer. The three that
/// matter most control its running summary: how fast older audio fades, how much extra weight the
/// current frame gets, and how much each frame is blended with the one before it. Defaults follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class RWKVTransducerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public RWKVTransducerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RWKVTransducerOptions(RWKVTransducerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // INHERITED FROM ModelOptions, AND THEREFORE EASY TO MISS. Every declared property is copied
        // below; Seed is not declared here, so it was silently dropped and a copied configuration
        // produced a DIFFERENT model from the one it was copied from -- the failure mode that costs
        // the most to diagnose, because the two configurations compare equal on everything visible.
        Seed = other.Seed;

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        CgmlpDim = other.CgmlpDim;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        TimeDecay = other.TimeDecay;
        CurrentTokenBonus = other.CurrentTokenBonus;
        TokenShiftMix = other.TokenShiftMix;
        BoundaryAware = other.BoundaryAware;
        Language = other.Language;
        // CLONED, NOT SHARED. `Vocabulary = other.Vocabulary` hands the copy the SAME array
        // instance, so a later write through either options object is seen by both -- the copy
        // constructor exists precisely to prevent that coupling, and for a reference type a bare
        // assignment does not. Null is preserved as null rather than becoming an empty array,
        // which would silently change "unset" into "set to nothing".
        Vocabulary = (string[])other.Vocabulary.Clone();
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 8;
    public int CgmlpDim { get; set; } = 3072;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets w, the per-step time-decay exponent of the RWKV recurrence.
    /// </summary>
    /// <remarks>
    /// Past frames are scaled by e^(-w) each step, so influence falls off geometrically instead of
    /// being truncated at a chunk boundary. Must be non-negative: a negative w would amplify the
    /// past every step and diverge over a long utterance.
    /// </remarks>
    /// <value>Defaults to 0.5.</value>
    public double TimeDecay { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets u, the extra weight the CURRENT frame carries relative to the decayed history.
    /// </summary>
    /// <value>Defaults to 1.0.</value>
    public double CurrentTokenBonus { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets mu, the token-shift mix: how much of the current frame the shifted input uses,
    /// the remainder coming from the previous frame. 1.0 disables the shift.
    /// </summary>
    /// <value>Defaults to 0.7.</value>
    public double TokenShiftMix { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets whether to use the boundary-aware transducer variant.
    /// </summary>
    /// <remarks>
    /// The paper evaluates both "RWKV-Transducer and RWKV-Boundary-Aware-Transducer". The
    /// boundary-aware arm adds an explicit emission-boundary signal; the plain transducer is the
    /// default.
    /// </remarks>
    /// <value>False by default.</value>
    public bool BoundaryAware { get; set; } = false;

    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };
}
