using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>Options for Efficient Conformer with progressive downsampling (Burchi &amp; Vielzeuf, 2021).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EfficientConformer model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class EfficientConformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EfficientConformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EfficientConformerOptions(EfficientConformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardExpansionFactor = other.FeedForwardExpansionFactor;
        ConvKernelSize = other.ConvKernelSize;
        InitialAttentionGroupSize = other.InitialAttentionGroupSize;
        DownsamplingFactor = other.DownsamplingFactor;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        ModelPath = other.ModelPath;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
        LearningRateFactor = other.LearningRateFactor;
        WeightDecay = other.WeightDecay;
        UseLayerNormalization = other.UseLayerNormalization;
        Language = other.Language;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Vocabulary = other.Vocabulary.ToArray();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the first-stage width. The paper's CTC-small model starts at 120.</summary>
    public int EncoderDim { get; set; } = 120;

    /// <summary>Gets or sets the encoder block count. CTC-small uses five blocks in each of three stages.</summary>
    public int NumEncoderLayers { get; set; } = 15;

    /// <summary>Gets or sets the attention head count used in all three stages.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the inner width of each feed-forward block as a multiple of the encoder width.</summary>
    public int FeedForwardExpansionFactor { get; set; } = 4;

    /// <summary>Gets or sets the Conformer depthwise-convolution kernel size.</summary>
    public int ConvKernelSize { get; set; } = 15;

    /// <summary>Gets or sets projected-frame attention grouping in the first encoder stage.</summary>
    public int InitialAttentionGroupSize { get; set; } = 3;

    /// <summary>Gets or sets the total time downsampling applied across the encoder stages: 1, 2, 4 or 8.</summary>
    public int DownsamplingFactor { get; set; } = 8;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 256;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the number of Transformer learning-rate warmup steps.</summary>
    /// <value>Defaults to 10,000 steps.</value>
    /// <remarks>All EfficientConformer CTC configurations in the authors' repository use a 10,000-step warmup.</remarks>
    public int WarmupSteps { get; set; } = 10000;

    /// <summary>Gets or sets the multiplicative factor for the Transformer learning-rate schedule.</summary>
    /// <value>Defaults to 2.0.</value>
    /// <remarks>The authors' CTC configurations set the schedule factor <c>K</c> to 2.</remarks>
    public double LearningRateFactor { get; set; } = 2.0;

    /// <summary>Gets or sets Adam's coupled L2 weight decay.</summary>
    /// <value>Defaults to 1e-6.</value>
    /// <remarks>The official EfficientConformer training configurations use Adam with weight decay 1e-6.</remarks>
    public double WeightDecay { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets whether the native encoder replaces its BatchNorm stages with
    /// LayerNorm. The research-default BatchNorm topology remains the default.
    /// </summary>
    public bool UseLayerNormalization { get; set; }
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Gets or sets the token strings, indexed by token id, that native transcripts are spelled with. Index 0 is the CTC blank and "|" separates words.</summary>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() =>
    [
        "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " "
    ];
}
