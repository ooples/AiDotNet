using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Canary-1B-Flash.</summary>
/// <remarks>
/// <para>The defaults: a Fast Conformer encoder of 32 blocks, width 1024, 8 heads, kernel 9, and a 4-layer Transformer decoder.</para>
/// <para><b>For Beginners:</b> These options set the size of the Conformer encoder, the feature front end and
/// the training augmentation.</para>
/// </remarks>
public class CanaryFlashOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CanaryFlashOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CanaryFlashOptions(CanaryFlashOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        ConvKernel = other.ConvKernel;
        RopeTheta = other.RopeTheta;
        NumDecoderLayers = other.NumDecoderLayers;
        DecoderFeedForwardDim = other.DecoderFeedForwardDim;
        MaxTextLength = other.MaxTextLength;
        SubsamplingChannels = other.SubsamplingChannels;
        FrequencyMasks = other.FrequencyMasks;
        FrequencyMaskWidth = other.FrequencyMaskWidth;
        TimeMasks = other.TimeMasks;
        TimeMaskRatio = other.TimeMaskRatio;
        VocabSize = other.VocabSize;
        Vocabulary = (string[])other.Vocabulary.Clone();
        Language = other.Language;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
    }

    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio a transcription covers, in seconds; longer input is truncated.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Log-mel bins (Canary: 128-dimensional features every 10 ms over 25 ms windows).</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Conformer width.</summary>
    public int EncoderDim { get; set; } = 1024;

    /// <summary>Conformer blocks.</summary>
    public int NumEncoderLayers { get; set; } = 32;

    /// <summary>Attention heads per block.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Expansion of each half-step feed-forward module (section 2.3: 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Depthwise convolution kernel (Fast Conformer, section 3.1: 9).</summary>
    public int ConvKernel { get; set; } = 9;

    /// <summary>Rotary base of the attention's relative position encoding, standing in for the paper's relative
    /// sinusoidal encoding (section 2.1).</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>Transformer decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 4;

    /// <summary>Feed-forward units of each decoder layer (4096).</summary>
    public int DecoderFeedForwardDim { get; set; } = 4096;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Channels of the 8x depthwise-separable subsampling (section 3.1: 256).</summary>
    public int SubsamplingChannels { get; set; } = 256;

    /// <summary>SpecAugment frequency masks (section 3.1: 2).</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins (section 3.1: F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks (section 3.1: 10).</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance (section 3.1: pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary size, blank (token 0) included (Canary: four 1024-token language tokenizers and 32 special tokens).</summary>
    public int VocabSize { get; set; } = 4128;

    /// <summary>The text of each token; tokens past the list decode as Unicode code points, and tokens in angle brackets
    /// decode to nothing.</summary>
    public string[] Vocabulary { get; set; } = { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    internal ConformerAsrConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, EncoderDim = EncoderDim, NumEncoderLayers = NumEncoderLayers,
        NumAttentionHeads = NumAttentionHeads, FeedForwardMultiplier = FeedForwardMultiplier, ConvKernel = ConvKernel, RopeTheta = RopeTheta,
        Decoder = ConformerAsrDecoder.Attention, AttentionDecoderLayers = NumDecoderLayers, AttentionDecoderFeedForward = DecoderFeedForwardDim, CtcWeight = 0, LabelSmoothing = 0, Subsampling = ConformerAsrSubsampling.DepthwiseStriding8x, SubsamplingChannels = SubsamplingChannels, FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
        VocabSize = VocabSize,
    };
}
