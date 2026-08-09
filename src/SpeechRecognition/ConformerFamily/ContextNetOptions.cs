using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>Options for ContextNet CNN encoder with squeeze-and-excitation (Han et al., 2020).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ContextNet model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class ContextNetOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ContextNetOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ContextNetOptions(ContextNetOptions other)
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
        NumBlocks = other.NumBlocks;
        NumSubBlocks = other.NumSubBlocks;
        KernelSize = other.KernelSize;
        WidthScaling = other.WidthScaling;
        SqueezeExcitationRatio = other.SqueezeExcitationRatio;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
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

    /// <summary>Nominal encoder width. Retained for callers and serialization round-trip.</summary>
    /// <remarks>
    /// The paper does not use a single uniform width: channels follow three groups (256 / 512 /
    /// 640, each scaled by <see cref="WidthScaling"/>). The native encoder builds those groups
    /// directly, so this value does not drive the block widths.
    /// </remarks>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Number of convolution blocks. The paper uses 23 (C0..C22).</summary>
    public int NumBlocks { get; set; } = 23;

    /// <summary>Number of convolution layers inside each interior block. The paper uses 5.</summary>
    public int NumSubBlocks { get; set; } = 5;

    /// <summary>Depthwise convolution kernel size. The paper uses 5.</summary>
    public int KernelSize { get; set; } = 5;

    /// <summary>Reduction ratio of the squeeze-and-excitation bottleneck.</summary>
    public int SqueezeExcitationRatio { get; set; } = 8;

    /// <summary>
    /// Width multiplier <c>alpha</c> applied to every channel group.
    /// </summary>
    /// <value>Defaults to 1.0 (the paper's medium model).</value>
    /// <remarks>
    /// <para>Han et al. 2020 evaluate alpha at 0.5, 1 and 2 for the small, medium and large
    /// ContextNet variants. Channel groups are 256/512/640 before scaling.</para>
    /// <para><b>For Beginners:</b> This is a single dial for how wide — and therefore how large
    /// and how slow — the whole model is. Halve it for a smaller, faster model; double it for a
    /// bigger, more accurate one.</para>
    /// </remarks>
    public double WidthScaling { get; set; } = 1.0;

    public int NumMels { get; set; } = 80;

    /// <summary>Output vocabulary size; must equal <see cref="Vocabulary"/>'s length.</summary>
    /// <value>The token count. Defaults to the built-in character tokenizer's size.</value>
    /// <remarks>
    /// <para>
    /// DERIVED FROM <see cref="Vocabulary"/>, not set independently. This defaulted to 1024 -- the
    /// paper's wordpiece configuration -- while the shipped tokenizer holds 34 characters. The model
    /// then built a 1024-class output layer that no tokenizer here could decode, so the documented
    /// default configuration could not produce text. There is no 1k wordpiece model in this
    /// repository to ship as the default; supply one through <see cref="Vocabulary"/> to train the
    /// paper's configuration, and this follows it.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different tokens the model can output, and it has
    /// to match the token list exactly -- otherwise the model emits IDs the decoder cannot name.</para>
    /// </remarks>
    public int VocabSize { get; set; } = GetDefaultVocabulary().Length;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";

    /// <summary>The token strings, indexed by token id.</summary>
    /// <value>The built-in character tokenizer by default: blank, four specials, a word separator, a-z, apostrophe and space.</value>
    /// <remarks>
    /// <para>Index 0 is the CTC blank, which the decoder never emits. Index 5 (<c>|</c>) is the word
    /// separator and decodes to a space.</para>
    /// <para><b>For Beginners:</b> The list of pieces the model can output, in id order. Replace it to
    /// use your own tokenizer, and set <see cref="VocabSize"/> to its length.</para>
    /// </remarks>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();

    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };

    /// <summary>Validates the configuration.</summary>
    /// <exception cref="InvalidOperationException">The vocabulary and its declared size disagree.</exception>
    /// <remarks>
    /// A size that does not match the token list is not recoverable at decode time: the output layer
    /// emits ids the vocabulary cannot name, and the only options left are to drop them silently or to
    /// fail somewhere far from the setting that caused it.
    /// </remarks>
    public void Validate()
    {
        if (Vocabulary is null || Vocabulary.Length == 0)
        {
            throw new InvalidOperationException(
                "ContextNetOptions.Vocabulary must contain at least one token.");
        }

        if (VocabSize != Vocabulary.Length)
        {
            throw new InvalidOperationException(
                $"ContextNetOptions.VocabSize is {VocabSize} but Vocabulary holds {Vocabulary.Length} " +
                "tokens. The output layer is sized from VocabSize and decoded through Vocabulary, so " +
                "they must agree. Set VocabSize = Vocabulary.Length, or supply a matching tokenizer.");
        }
    }
}
