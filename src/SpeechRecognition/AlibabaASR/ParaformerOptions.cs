using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for Paraformer: fast and accurate parallel Transformer for non-autoregressive ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Paraformer model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class ParaformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ParaformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ParaformerOptions(ParaformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        // Copy the array itself, not the reference: a clone that shares the original's vocabulary
        // would see later edits to it, which is not what copying an options object should mean.
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>Gets or sets the parallel decoder width.</summary>
    /// <value>Defaults to 512, preserving the existing Paraformer helper architecture.</value>
    public int DecoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of non-autoregressive Transformer decoder blocks.</summary>
    /// <value>Defaults to 6, preserving the existing Paraformer helper architecture.</value>
    public int NumDecoderLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>Gets or sets the feed-forward inner width used by encoder and decoder blocks.</summary>
    /// <value>Defaults to the paper's 2048 hidden units.</value>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 8404;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Gets or sets the AdamW learning rate used by native training.</summary>
    /// <value>Defaults to 1e-3, preserving the model's previous optimizer behavior.</value>
    public double LearningRate { get; set; } = 1e-3;
    /// <summary>Gets or sets the decoupled AdamW weight decay used by native training.</summary>
    /// <value>Defaults to 0.01, preserving the framework AdamW default.</value>
    public double WeightDecay { get; set; } = 0.01;
    /// <summary>Optional vocabulary mapping token IDs to text tokens (e.g., SentencePiece vocabulary).</summary>
    public string[]? Vocabulary { get; set; }
}
