using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for SenseVoice: multi-task speech understanding model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure SenseVoice-Small. Supported defaults follow
/// the released FunASR configuration: 512-wide SAN-M encoder, 50 + 20 blocks, 4 attention heads,
/// 2048-wide feed-forward layers, FSMN kernel 11, low frame rate 7/6, 25,055 tokens, 80 mel bands, and AdamW at 2e-5.</para>
/// </remarks>
public class SenseVoiceOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SenseVoiceOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SenseVoiceOptions(SenseVoiceOptions other)
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
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        FeedForwardDim = other.FeedForwardDim;
        NumTpEncoderLayers = other.NumTpEncoderLayers;
        FsmnKernelSize = other.FsmnKernelSize;
        LfrM = other.LfrM;
        LfrN = other.LfrN;
        UseInverseTextNormalization = other.UseInverseTextNormalization;
        TargetsIncludeRichTokens = other.TargetsIncludeRichTokens;
        Vocabulary = (string[])other.Vocabulary.Clone();
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 25055;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>The language query: auto (the release's default), zh, en, yue, ja, ko or nospeech.</summary>
    public string Language { get; set; } = "auto";
    /// <summary>Gets or sets the hidden width of the feed-forward network. Defaults to <c>2048</c>.</summary>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>SAN-M blocks after the first norm (<c>tp_blocks</c>, 20).</summary>
    public int NumTpEncoderLayers { get; set; } = 20;
    /// <summary>The FSMN memory's kernel (11).</summary>
    public int FsmnKernelSize { get; set; } = 11;
    /// <summary>Filterbank frames stacked by the low frame rate front end (7).</summary>
    public int LfrM { get; set; } = 7;
    /// <summary>The stride between stacks (6, 60 ms).</summary>
    public int LfrN { get; set; } = 6;
    /// <summary>The text-normalization query: with inverse text normalization (14) or without (15, the release's
    /// default).</summary>
    public bool UseInverseTextNormalization { get; set; } = false;
    /// <summary>Training targets carry the four rich tokens (language, emotion, event, text normalization) before the
    /// transcript, as the release's do; their cross-entropy is then added to the CTC loss.</summary>
    public bool TargetsIncludeRichTokens { get; set; } = false;
    /// <summary>Token strings by id; ids past it decode as Unicode code points.</summary>
    public string[] Vocabulary { get; set; } = Array.Empty<string>();
    /// <summary>Gets or sets the optimizer learning rate. Defaults to <c>2e-5</c>.</summary>
    public double LearningRate { get; set; } = 2e-5;
    /// <summary>Gets or sets the weight decay of the default optimizer.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
