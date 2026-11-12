using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>Options for Conformer.</summary>
/// <remarks>
/// <para>The defaults: 16 Conformer blocks, width 256, 4 heads, a 32-wide depthwise kernel, a 640-unit single-layer LSTM prediction network and a 640-wide joint network.</para>
/// <para><b>For Beginners:</b> These options set the size of the Conformer encoder and decoder, the feature front end and
/// the training augmentation.</para>
/// </remarks>
public class ConformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ConformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ConformerOptions(ConformerOptions other)
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
        PredictionDim = other.PredictionDim;
        PredictionLayers = other.PredictionLayers;
        JointDim = other.JointDim;
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
    public int NumEncoderLayers { get; set; } = 16;

    /// <summary>Attention heads per block.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Expansion of each half-step feed-forward module (section 2.3: 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Depthwise convolution kernel (Table 1: 32).</summary>
    public int ConvKernel { get; set; } = 32;

    /// <summary>Rotary base of the attention's relative position encoding, standing in for the paper's relative
    /// sinusoidal encoding (section 2.1).</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>LSTM units of the prediction network (section 3.1: a single-layer LSTM of 640 units for Conformer M and L,
    /// 320 for S).</summary>
    public int PredictionDim { get; set; } = 640;

    /// <summary>LSTM layers of the prediction network (section 3.1: 1).</summary>
    public int PredictionLayers { get; set; } = 1;

    /// <summary>Width of the joint network.</summary>
    public int JointDim { get; set; } = 640;

    /// <summary>SpecAugment frequency masks (section 3.1: 2).</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins (section 3.1: F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks (section 3.1: 10).</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance (section 3.1: pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary size, blank (token 0) included.</summary>
    public int VocabSize { get; set; } = 34;

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
        Decoder = ConformerAsrDecoder.Transducer, PredictionDim = PredictionDim, PredictionLayers = PredictionLayers, JointDim = JointDim,
        FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
        VocabSize = VocabSize,
    };
}
