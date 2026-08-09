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
        LearningRate = other.LearningRate;
        NumAttentionHeads = other.NumAttentionHeads;
        CgmlpDim = other.CgmlpDim;
        MaxEncoderFrames = other.MaxEncoderFrames;
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
    /// <summary>
    /// Gets or sets the number of encoder blocks. Default 18, the paper's value.
    /// </summary>
    /// <remarks>
    /// An &amp; Zhang (arXiv:2309.14758) give "Encoder Blocks N: 18" for BOTH the small and the large
    /// configuration -- the two differ in width, not depth. This was 12, which is the Conformer family's
    /// convention rather than this paper's, and depth is the axis the paper holds fixed.
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 18;
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>Width of the convolutional gating MLP branch.</summary>
    /// <value>The cgMLP width. Default 3072.</value>
    /// <remarks>
    /// Retained for callers supplying their own Branchformer-style <c>Architecture.Layers</c>. The
    /// native RWKV encoder has no cgMLP branch -- its blocks are time mixing plus channel mixing -- so
    /// it does not read this.
    /// </remarks>
    public int CgmlpDim { get; set; } = 3072;

    /// <summary>Longest encoder frame count the RWKV blocks are built for.</summary>
    /// <value>The frame ceiling. Default 750, matching 30 s of audio after stride-4 subsampling at 10 ms hops.</value>
    /// <remarks>
    /// <para>
    /// <see cref="RWKVLayer{T}"/> takes a sequence length at construction, so the encoder needs one.
    /// Its recurrent state is constant-size regardless, which is the whole point of the architecture;
    /// this bounds the layer's own buffers, not the memory used per frame.
    /// </para>
    /// <para><b>For Beginners:</b> The longest utterance the encoder is sized for. Raise it if you
    /// transcribe recordings longer than <see cref="MaxAudioLengthSeconds"/>.</para>
    /// </remarks>
    public int MaxEncoderFrames { get; set; } = 750;
    public int NumMels { get; set; } = 80;
    /// <summary>Output vocabulary size; must equal <see cref="Vocabulary"/>'s length.</summary>
    /// <value>The token count. Defaults to the built-in character tokenizer's size.</value>
    /// <remarks>
    /// This defaulted to 5000 while the shipped tokenizer holds 34 characters, so the model built a
    /// 5000-class output layer emitting ids no decoder here could name. Supply a matching
    /// <see cref="Vocabulary"/> to use a larger tokenizer; <c>Validate</c> requires the two to agree.
    /// </remarks>
    public int VocabSize { get; set; } = GetDefaultVocabulary().Length;
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

    /// <summary>
    /// Gets or sets the AdamW learning rate used when no optimizer is injected.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This property did not exist, and the optimizer was constructed bare.</b> That broke both
    /// halves of the rule at once: the model trained at AdamW's own generic default rather than a rate
    /// chosen for it, and a caller had no way to correct that short of building and injecting a whole
    /// optimizer.
    /// </para>
    /// <para>
    /// <b>The paper does not state a learning rate.</b> An &amp; Zhang (arXiv:2309.14758) specify the
    /// architecture -- 18 encoder blocks, d_io 512/640 -- and the BAT training objective, but no
    /// optimizer settings. Rather than invent a number and label it the paper's, the default is 1e-3,
    /// the value the rest of this repository's streaming-ASR encoders train at, and it is documented
    /// here as a convention rather than a citation so nobody later "corrects" it against a paper
    /// section that does not exist.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };

    /// <summary>Validates the configuration.</summary>
    /// <exception cref="InvalidOperationException">The vocabulary and its declared size disagree.</exception>
    public void Validate()
    {
        if (Vocabulary is null || Vocabulary.Length == 0)
        {
            throw new InvalidOperationException(
                "RWKVTransducerOptions.Vocabulary must contain at least one token.");
        }

        if (VocabSize != Vocabulary.Length)
        {
            throw new InvalidOperationException(
                $"RWKVTransducerOptions.VocabSize is {VocabSize} but Vocabulary holds {Vocabulary.Length} " +
                "tokens. The output layer is sized from VocabSize and decoded through Vocabulary, so " +
                "they must agree.");
        }
    }
}
