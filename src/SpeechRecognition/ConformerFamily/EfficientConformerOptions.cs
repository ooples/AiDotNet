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
        FeedForwardExpansionFactor = other.FeedForwardExpansionFactor;
        DownsamplingFactor = other.DownsamplingFactor;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
        LearningRateFactor = other.LearningRateFactor;
        WeightDecay = other.WeightDecay;
        UseLayerNormalization = other.UseLayerNormalization;
        Language = other.Language;
        Vocabulary = other.Vocabulary.ToArray();
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 16;
    public int NumAttentionHeads { get; set; } = 8;
    public int FeedForwardExpansionFactor { get; set; } = 4;
    public int DownsamplingFactor { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
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
    public string Language { get; set; } = "en";
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();
    private static string[] GetDefaultVocabulary() => new[] { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };
}
