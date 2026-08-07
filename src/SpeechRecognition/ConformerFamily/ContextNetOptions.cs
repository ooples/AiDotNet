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
        Vocabulary = other.Vocabulary;
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

    /// <summary>Output vocabulary size. The paper uses a 1k wordpiece model.</summary>
    /// <remarks>
    /// Previously defaulted to 5000, which does not correspond to any configuration in the paper.
    /// </remarks>
    public int VocabSize { get; set; } = 1024;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };
}
