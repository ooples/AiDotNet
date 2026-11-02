using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// Configuration options for the E-Branchformer (Enhanced Branchformer) speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// E-Branchformer (Kim et al., 2022) improves on Branchformer with an enhanced merge module
/// that uses depthwise convolution for better local-global fusion, achieving SOTA on LibriSpeech
/// (WER 2.1%/4.2% test-clean/other) with the ESPnet toolkit.
/// </para>
/// </remarks>
public class EBranchformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EBranchformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EBranchformerOptions(EBranchformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        CgmlpDim = other.CgmlpDim;
        MergeDim = other.MergeDim;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        Vocabulary = other.Vocabulary;
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>Gets or sets the inner width of the convolutional gating MLP (cgMLP) branch.</summary>
    public int CgmlpDim { get; set; } = 3072;
    /// <summary>Gets or sets the merge-module width of the paper's configuration. The native layers merge the two branches at the encoder width and do not read it.</summary>
    public int MergeDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 5000;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Gets or sets the token strings, indexed by token id, that native transcripts are spelled with. Index 0 is the CTC blank and "|" separates words.</summary>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();

    private static string[] GetDefaultVocabulary() => new[]
    {
        "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "'", " "
    };
}
