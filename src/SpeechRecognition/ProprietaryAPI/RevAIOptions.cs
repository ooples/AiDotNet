using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>Options for Rev AI's Reverb ASR.</summary>
/// <remarks>
/// <para>The defaults: 18 Conformer blocks, a CTC layer and a bidirectional attention decoder of three left-to-right and
/// three right-to-left layers, about 280M parameters (Reverb report, section 2.2). The report gives the layer counts and
/// the total; the width (640, 8 heads, feed-forward 2,560) is the one that totals about 280M with those layers, and the
/// kernel, CTC weight, reverse weight and label smoothing are WeNet's U2++ defaults.</para>
/// <para><b>For Beginners:</b> These options set the size of the Conformer encoder, the feature front end and
/// the training augmentation.</para>
/// </remarks>
public class RevAIOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public RevAIOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RevAIOptions(RevAIOptions other)
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
        Seed = other.Seed;
        AttentionDecoderLayers = other.AttentionDecoderLayers;
        ReverseDecoderLayers = other.ReverseDecoderLayers;
        ReverseWeight = other.ReverseWeight;
        AttentionDecoderFeedForward = other.AttentionDecoderFeedForward;
        CtcWeight = other.CtcWeight;
        LabelSmoothing = other.LabelSmoothing;
        MaxTextLength = other.MaxTextLength;
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

    /// <summary>Conformer width (640).</summary>
    public int EncoderDim { get; set; } = 640;

    /// <summary>Conformer blocks (18, section 2.2).</summary>
    public int NumEncoderLayers { get; set; } = 18;

    /// <summary>Attention heads per block.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Expansion of each half-step feed-forward module (section 2.3: 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Depthwise convolution kernel (15, WeNet's <c>cnn_module_kernel</c>).</summary>
    public int ConvKernel { get; set; } = 15;

    /// <summary>Rotary base of the attention's relative position encoding, standing in for WeNet's relative
    /// positional encoding.</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>Left-to-right decoder layers (3, section 2.2).</summary>
    public int AttentionDecoderLayers { get; set; } = 3;

    /// <summary>Right-to-left decoder layers (3, section 2.2).</summary>
    public int ReverseDecoderLayers { get; set; } = 3;

    /// <summary>The right-to-left decoder's share of the attention loss (0.3, WeNet U2++'s <c>reverse_weight</c>).</summary>
    public double ReverseWeight { get; set; } = 0.3;

    /// <summary>Feed-forward units of each decoder layer (2,560).</summary>
    public int AttentionDecoderFeedForward { get; set; } = 2560;

    /// <summary>Weight w of the CTC loss in the joint objective w L_CTC + (1 - w) L_attention.</summary>
    public double CtcWeight { get; set; } = 0.3;

    /// <summary>Label smoothing of the decoder's cross-entropy.</summary>
    public double LabelSmoothing { get; set; } = 0.1;

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
        NumAttentionHeads = NumAttentionHeads, FeedForwardMultiplier = FeedForwardMultiplier, ConvKernel = ConvKernel, RopeTheta = RopeTheta,
        AttentionDecoderLayers = AttentionDecoderLayers, ReverseDecoderLayers = ReverseDecoderLayers, ReverseWeight = ReverseWeight, AttentionDecoderFeedForward = AttentionDecoderFeedForward, CtcWeight = CtcWeight, LabelSmoothing = LabelSmoothing, FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
        VocabSize = VocabSize,
    };
}
