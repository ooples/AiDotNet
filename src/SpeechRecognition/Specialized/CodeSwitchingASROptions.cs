using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// Options for code-switching ASR, following Zeng et al., "On the End-to-End Solution to
/// Mandarin-English Code-switching Speech Recognition" (Interspeech 2019, arXiv:1811.00241).
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the paper: a hybrid CTC/attention objective with a CTC weight of 0.2, an
/// auxiliary language-identification task weighted at 0.2, and a 3k BPE inventory for English
/// alongside Mandarin characters.
/// </para>
/// <para>
/// <b>For Beginners:</b> Code-switching is when a speaker mixes languages inside one sentence.
/// These settings control three things the model does about that: how it balances two different
/// ways of reading speech (CTC and attention), how much effort it spends explicitly working out
/// WHICH language it is hearing at each moment, and how the two languages' writing systems share
/// the output alphabet.
/// </para>
/// </remarks>
public class CodeSwitchingASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CodeSwitchingASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CodeSwitchingASROptions(CodeSwitchingASROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        _explicitVocabSize = other._explicitVocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        CtcWeight = other.CtcWeight;
        LidWeight = other.LidWeight;
        DecoderDim = other.DecoderDim;
        NumDecoderLayers = other.NumDecoderLayers;
        EnglishBpeVocabSize = other.EnglishBpeVocabSize;
        MandarinCharVocabSize = other.MandarinCharVocabSize;
        SharedLidAttention = other.SharedLidAttention;

        // VocabSize IS NOT ASSIGNED HERE, DELIBERATELY. _explicitVocabSize is copied above, which
        // carries BOTH of this property's states exactly: explicitly set, or unset and therefore
        // derived from MandarinCharVocabSize + EnglishBpeVocabSize + 1.
        //
        // Going through the public setter instead -- `VocabSize = other.VocabSize` -- read the
        // GETTER, which materializes the derived value, and then stored it as an EXPLICIT one. A
        // clone of a derived-size options object came back pinned: change MandarinCharVocabSize on
        // both afterwards and the original recomputes while the clone does not. The remarks on
        // VocabSize record that this explicit/derived distinction has already caused one real
        // defect, so collapsing it in the copy constructor is the same failure a second time.
    }

    /// <summary>
    /// Gets or sets the CTC branch's weight in the joint objective
    /// <c>L = (1 - CtcWeight) * L_att + CtcWeight * L_ctc + LidWeight * L_lid</c>.
    /// </summary>
    /// <remarks>
    /// The paper fixes the CTC weight at 0.2, i.e. the attention branch carries 0.8. CTC's monotonic
    /// alignment stabilizes the attention decoder early in training; attention supplies the
    /// context-dependence CTC's conditional-independence assumption cannot.
    /// </remarks>
    /// <value>Defaults to 0.2 (the paper's value).</value>
    public double CtcWeight { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets lambda_2, the weight of the auxiliary language-identification loss.
    /// </summary>
    /// <remarks>
    /// This is the paper's central contribution: "we propose a multitask learning recipe, where a
    /// language identification task is EXPLICITLY learned in addition to the E2E speech recognition
    /// task". Setting it to zero removes the contribution and leaves a plain hybrid CTC/attention
    /// model, which is the paper's own baseline.
    /// </remarks>
    /// <value>Defaults to 0.2, the paper's tuned optimum.</value>
    public double LidWeight { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether the LID head shares the ASR decoder's attention (the paper's
    /// <c>LID_shared</c>) rather than using its own (<c>LID_indep</c>).
    /// </summary>
    /// <value>Defaults to true (<c>LID_shared</c>).</value>
    public bool SharedLidAttention { get; set; } = true;

    /// <summary>Gets or sets the attention decoder's hidden width.</summary>
    /// <value>Defaults to 320 (the paper's value).</value>
    public int DecoderDim { get; set; } = 320;

    /// <summary>Gets or sets the number of attention decoder layers.</summary>
    /// <value>Defaults to 1 (the paper's value).</value>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of English BPE subword units.
    /// </summary>
    /// <remarks>
    /// The paper trains BPE inventories of 1.9k, 2k, 3k and 4k and finds 3k best.
    /// </remarks>
    /// <value>Defaults to 3000 (the paper's best).</value>
    public int EnglishBpeVocabSize { get; set; } = 3000;

    /// <summary>
    /// Gets or sets the number of Mandarin character units.
    /// </summary>
    /// <remarks>
    /// The two languages do NOT share one inventory: "Mandarin uses characters while English uses
    /// BPE units". The output alphabet is the concatenation of the two, which is what makes a token
    /// identify its own language and gives the LID task something to be consistent with.
    /// </remarks>
    /// <value>Defaults to 3000.</value>
    public int MandarinCharVocabSize { get; set; } = 3000;

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Gets or sets the encoder's hidden width.</summary>
    /// <value>Defaults to 320 (the paper's BLSTM width).</value>
    public int EncoderDim { get; set; } = 320;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    /// <value>Defaults to 6 (the paper's BLSTM depth).</value>
    public int NumEncoderLayers { get; set; } = 6;

    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;

    private int? _explicitVocabSize;

    /// <summary>
    /// Gets or sets the total output inventory. By DEFAULT this is derived as Mandarin characters
    /// plus English BPE units plus one CTC blank, because the paper's point is that the alphabet is
    /// a CONCATENATION of two per-language inventories rather than an independently chosen unified
    /// size.
    /// </summary>
    /// <remarks>
    /// Setting it explicitly overrides the derivation — needed by callers that size the alphabet
    /// directly, such as bounded test fixtures. An earlier revision made this derived with a no-op
    /// setter, which SILENTLY discarded those callers' values and left the model building a
    /// 6001-wide output against a 32-wide fixture. When it is set explicitly the two-inventory
    /// boundary is still well-defined, via <see cref="MandarinTokenCount"/>.
    /// </remarks>
    public int VocabSize
    {
        get => _explicitVocabSize ?? (MandarinCharVocabSize + EnglishBpeVocabSize + 1);
        set => _explicitVocabSize = value;
    }

    /// <summary>
    /// The number of Mandarin token ids, i.e. where the English inventory begins (after the blank).
    /// </summary>
    /// <remarks>
    /// When <see cref="VocabSize"/> has been set explicitly the configured per-language sizes may
    /// not fit inside it, so the boundary is placed at half the available ids. The two inventories
    /// stay disjoint either way, which is the property the LID target depends on.
    /// </remarks>
    public int MandarinTokenCount =>
        _explicitVocabSize is int v
            ? Math.Max(1, (v - 1) / 2)
            : MandarinCharVocabSize;

    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
