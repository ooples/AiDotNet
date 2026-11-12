using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>Options for the nearest disclosed architecture to Azure Speech, Microsoft's streaming Conformer transducer (Chen et al. 2021).</summary>
/// <remarks>
/// <para>See <see cref="AzureSpeechSTT{T}"/> for which defaults the vendor or paper discloses and which the stand-in infers.</para>
/// <para><b>For Beginners:</b> These options set the size of the encoder and decoder, the feature front end and the training
/// augmentation.</para>
/// </remarks>
public class AzureSpeechSTTOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public AzureSpeechSTTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AzureSpeechSTTOptions(AzureSpeechSTTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardUnits = other.FeedForwardUnits;
        ConvKernel = other.ConvKernel;
        RopeTheta = other.RopeTheta;
        AttentionChunkSize = other.AttentionChunkSize;
        AttentionLeftChunks = other.AttentionLeftChunks;
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
        Seed = other.Seed;
    }

    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio a transcription covers, in seconds; longer input is truncated.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Log-mel bins (80 over 25 ms windows with a 10 ms stride).</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Conformer width (section 4.1: 640).</summary>
    public int EncoderDim { get; set; } = 640;

    /// <summary>Conformer blocks (18, the depth section 4.1 gives the Transformer transducer C-T is matched to).</summary>
    public int NumEncoderLayers { get; set; } = 18;

    /// <summary>Attention heads per block (not stated by the paper).</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Hidden units of each feed-forward module (1,024, the Transformer transducer's, section 4.1).</summary>
    public int FeedForwardUnits { get; set; } = 1024;

    /// <summary>Depthwise convolution kernel (section 4.1: 3).</summary>
    public int ConvKernel { get; set; } = 3;

    /// <summary>Rotary base standing in for the paper's relative position encoding.</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>Encoder frames (40 ms each) per attention chunk: 720 ms, the paper's 24 frames of 30 ms (section 4.2.2).</summary>
    public int AttentionChunkSize { get; set; } = 18;

    /// <summary>Chunks of history each chunk also attends to: 2.16 s, the nearest whole number of chunks above the paper's 60 frames (1.8 s).</summary>
    public int AttentionLeftChunks { get; set; } = 3;

    /// <summary>LSTM units of the prediction network (section 4.1: 720).</summary>
    public int PredictionDim { get; set; } = 720;

    /// <summary>LSTM layers of the prediction network (section 4.1: 2).</summary>
    public int PredictionLayers { get; set; } = 2;

    /// <summary>Width of the joint network (the encoder width; not stated).</summary>
    public int JointDim { get; set; } = 640;

    /// <summary>SpecAugment frequency masks (Park et al. 2019; Gulati et al. 2020, section 3.1: 2).</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins (F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks (10).</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance (pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary size, blank (token 0) included: 4,000 sentence pieces (section 4.1) and the blank.</summary>
    public int VocabSize { get; set; } = 4001;

    /// <summary>The text of each token; tokens past the list decode as Unicode code points, and tokens in angle brackets decode to nothing.</summary>
    public string[] Vocabulary { get; set; } = { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    internal ConformerAsrConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, EncoderDim = EncoderDim, NumEncoderLayers = NumEncoderLayers, NumAttentionHeads = NumAttentionHeads,
        Decoder = ConformerAsrDecoder.Transducer, FeedForwardUnits = FeedForwardUnits, AttentionChunkSize = AttentionChunkSize, AttentionLeftChunks = AttentionLeftChunks, PredictionDim = PredictionDim, PredictionLayers = PredictionLayers, JointDim = JointDim, ConvKernel = ConvKernel, RopeTheta = RopeTheta,
        FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio, VocabSize = VocabSize,
    };
}
