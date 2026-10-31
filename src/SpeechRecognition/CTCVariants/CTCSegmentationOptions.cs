using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.CTCVariants;

/// <summary>Options for CTC-segmentation.</summary>
/// <remarks>
/// <para>The defaults: a 12-layer Transformer encoder, width 256, 4 heads, a 6-layer decoder and a CTC layer.</para>
/// <para><b>For Beginners:</b> These options set the size of the Transformer encoder, the feature front end and
/// the training augmentation.</para>
/// </remarks>
public class CTCSegmentationOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CTCSegmentationOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CTCSegmentationOptions(CTCSegmentationOptions other)
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
        RopeTheta = other.RopeTheta;
        AttentionDecoderLayers = other.AttentionDecoderLayers;
        AttentionDecoderFeedForward = other.AttentionDecoderFeedForward;
        CtcWeight = other.CtcWeight;
        LabelSmoothing = other.LabelSmoothing;
        MaxTextLength = other.MaxTextLength;
        ConfidenceWindow = other.ConfidenceWindow;
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

    /// <summary>Log-mel bins (section 3.1: 80-channel filterbanks over 25 ms windows with a 10 ms stride).</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Conformer width.</summary>
    public int EncoderDim { get; set; } = 256;

    /// <summary>Conformer blocks.</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Attention heads per block.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Expansion of each feed-forward network: 2048 units at width 256.</summary>
    public int FeedForwardMultiplier { get; set; } = 8;


    /// <summary>Rotary base of the attention; zero, since the Transformer adds sinusoidal positions.</summary>
    public double RopeTheta { get; set; } = 0.0;


    /// <summary>Transformer decoder layers trained jointly with the CTC layer.</summary>
    public int AttentionDecoderLayers { get; set; } = 6;

    /// <summary>Feed-forward units of each decoder layer.</summary>
    public int AttentionDecoderFeedForward { get; set; } = 2048;

    /// <summary>Weight w of the CTC loss in the joint objective w L_CTC + (1 - w) L_attention.</summary>
    public double CtcWeight { get; set; } = 0.3;

    /// <summary>Label smoothing of the decoder's cross-entropy.</summary>
    public double LabelSmoothing { get; set; } = 0.1;

    /// <summary>Frames per window of the segment confidence (Equation 3's L).</summary>
    public int ConfidenceWindow { get; set; } = 30;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 256;

    /// <summary>SpecAugment frequency masks (section 3.1: 2).</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins (section 3.1: F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks (section 3.1: 10).</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance (section 3.1: pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary size, blank (token 0) included.</summary>
    public int VocabSize { get; set; } = 5000;

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
        NumAttentionHeads = NumAttentionHeads, FeedForwardMultiplier = FeedForwardMultiplier, RopeTheta = RopeTheta,
        Block = ConformerAsrBlock.Transformer, AttentionDecoderLayers = AttentionDecoderLayers, AttentionDecoderFeedForward = AttentionDecoderFeedForward, CtcWeight = CtcWeight, LabelSmoothing = LabelSmoothing, FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
        VocabSize = VocabSize,
    };
}
