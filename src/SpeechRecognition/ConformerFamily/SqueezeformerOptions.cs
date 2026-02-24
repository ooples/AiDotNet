using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// Configuration options for the Squeezeformer speech recognition model.
/// </summary>
/// <remarks>
/// <para>Squeezeformer (Kim et al., 2022) improves Conformer with a temporal U-Net structure,
/// micro-macro design (pre-norm instead of post-norm), and depthwise separable downsampling
/// for efficient computation with better accuracy.</para>
/// </remarks>
public class SqueezeformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SqueezeformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SqueezeformerOptions(SqueezeformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardExpansionFactor = other.FeedForwardExpansionFactor;
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
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 16;
    public int NumAttentionHeads { get; set; } = 8;
    public int FeedForwardExpansionFactor { get; set; } = 4;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };
}
