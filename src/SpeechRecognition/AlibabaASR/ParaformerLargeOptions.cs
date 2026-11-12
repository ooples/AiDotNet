using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for Paraformer-large: the released 220M-parameter Mandarin Paraformer.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ParaformerLarge model. Defaults are the release's config.yaml: a 50-block SAN-M encoder of width 512, a 16-block SAN-M decoder, the convolutional CIF predictor, glancing ratio 0.75 and 8,404 characters.</para>
/// </remarks>
public class ParaformerLargeOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ParaformerLargeOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ParaformerLargeOptions(ParaformerLargeOptions other)
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
        NumMels = other.NumMels;
        LfrM = other.LfrM;
        LfrN = other.LfrN;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        FsmnKernelSize = other.FsmnKernelSize;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        LabelSmoothing = other.LabelSmoothing;
        SamplingRatio = other.SamplingRatio;
        PredictorWeight = other.PredictorWeight;
        TailThreshold = other.TailThreshold;
        WarmupSteps = other.WarmupSteps;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Kaldi filterbank bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Filterbank frames stacked by the low frame rate front end (7).</summary>
    public int LfrM { get; set; } = 7;
    /// <summary>The stride between stacks (6).</summary>
    public int LfrN { get; set; } = 6;
    /// <summary>The encoder's width (512), which the predictor and the decoder share.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>SAN-M encoder blocks (50).</summary>
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>SAN-M decoder blocks with cross-attention (16), before the final feed-forward block.</summary>
    public int NumDecoderLayers { get; set; } = 16;
    /// <summary>Gets or sets the number of attention heads. Defaults to <c>4</c>.</summary>
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>Gets or sets the hidden width of the feed-forward network. Defaults to <c>2048</c>.</summary>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>The FSMN memories' kernel (11).</summary>
    public int FsmnKernelSize { get; set; } = 11;
    /// <summary>Output vocabulary, blank (0), start (1) and end (2) included (8,404).</summary>
    public int VocabSize { get; set; } = 8404;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"zh"</c>.</summary>
    public string Language { get; set; } = "zh";
    /// <summary>The decoder loss's label smoothing (0.1).</summary>
    public double LabelSmoothing { get; set; } = 0.1;
    /// <summary>The glancing sampler's λ (0.75).</summary>
    public double SamplingRatio { get; set; } = 0.75;
    /// <summary>The MAE token-count loss's weight (1).</summary>
    public double PredictorWeight { get; set; } = 1.0;
    /// <summary>The CIF tail weight at inference (0.45).</summary>
    public double TailThreshold { get; set; } = 0.45;
    /// <summary>The learning rate's warmup steps (30,000, FunASR's warmuplr).</summary>
    public int WarmupSteps { get; set; } = 30000;
    /// <summary>Optional vocabulary mapping token IDs to text tokens (the release's tokens.json).</summary>
    public string[]? Vocabulary { get; set; }
}
