using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Nemotron Speech Streaming: a cache-aware Fast Conformer with an RNN-T decoder.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released nvidia/nemotron-speech-streaming-en-0.6b: 128-band log-mel
/// features (not normalized, since a stream has no whole-utterance statistics), 8x causal depthwise-separable
/// subsampling to 80 ms frames, 24 Conformer blocks of width 1,024 with 8 heads and a causal kernel-9 convolution, each
/// frame attending within its chunk and the 70 frames of chunks before it, and an RNN-T decoder (a two-layer 640-unit
/// LSTM prediction network and a 640-wide joint network) over a 1,024-piece SentencePiece vocabulary.</para>
/// </remarks>
public class NemotronSpeechOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public NemotronSpeechOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public NemotronSpeechOptions(NemotronSpeechOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        ConvKernel = other.ConvKernel;
        SubsamplingChannels = other.SubsamplingChannels;
        LeftContextFrames = other.LeftContextFrames;
        LookaheadFrames = other.LookaheadFrames;
        RopeTheta = other.RopeTheta;
        PredictionDim = other.PredictionDim;
        PredictionLayers = other.PredictionLayers;
        JointDim = other.JointDim;
        FastEmitLambda = other.FastEmitLambda;
        FrequencyMasks = other.FrequencyMasks;
        FrequencyMaskWidth = other.FrequencyMaskWidth;
        TimeMasks = other.TimeMasks;
        TimeMaskRatio = other.TimeMaskRatio;
        VocabSize = other.VocabSize;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
        WarmupSteps = other.WarmupSteps;
        NoamFactor = other.NoamFactor;
        WeightDecay = other.WeightDecay;
        Language = other.Language;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 60;
    /// <summary>Log-mel bands (128).</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>The encoder's width (1,024).</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>The encoder's Conformer blocks (24).</summary>
    public int NumEncoderLayers { get; set; } = 24;
    /// <summary>The encoder's attention heads (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>The feed-forward expansion (4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;
    /// <summary>The causal depthwise convolution kernel (9).</summary>
    public int ConvKernel { get; set; } = 9;
    /// <summary>Channels of the subsampling convolutions (256).</summary>
    public int SubsamplingChannels { get; set; } = 256;
    /// <summary>Encoder frames of left context (70, the release's <c>att_context_size[0]</c>): each chunk attends to
    /// the ⌊70 / chunk⌋ chunks before it.</summary>
    public int LeftContextFrames { get; set; } = 70;
    /// <summary>Encoder frames of lookahead (13; the release supports 0, 1, 6 and 13): a chunk is this many frames plus
    /// one, so 13 is 1.12 s.</summary>
    public int LookaheadFrames { get; set; } = 13;
    /// <summary>Rotary base standing in for the release's relative positional encoding.</summary>
    public double RopeTheta { get; set; } = 10000.0;
    /// <summary>The prediction network's LSTM width (640).</summary>
    public int PredictionDim { get; set; } = 640;
    /// <summary>The prediction network's LSTM layers (2).</summary>
    public int PredictionLayers { get; set; } = 2;
    /// <summary>The joint network's width (640).</summary>
    public int JointDim { get; set; } = 640;
    /// <summary>FastEmit regularization of the transducer loss (0.005).</summary>
    public double FastEmitLambda { get; set; } = 0.005;
    public int FrequencyMasks { get; set; } = 2;
    public int FrequencyMaskWidth { get; set; } = 27;
    public int TimeMasks { get; set; } = 10;
    public double TimeMaskRatio { get; set; } = 0.05;
    /// <summary>Output symbols, the blank (0) first: the 1,024 SentencePiece pieces follow as ids 1 to 1,024 (the
    /// release puts its blank last, at 1,024).</summary>
    public int VocabSize { get; set; } = 1025;
    /// <summary>Optional vocabulary mapping output ids to text in place of the release's SentencePiece pieces.</summary>
    public string[]? Vocabulary { get; set; }
    /// <summary>The Noam schedule's warm-up steps (8,000).</summary>
    public int WarmupSteps { get; set; } = 8000;
    /// <summary>The Noam schedule's factor (the release's <c>lr: 3.0</c>), scaled by the encoder width's inverse square
    /// root.</summary>
    public double NoamFactor { get; set; } = 3.0;
    /// <summary>AdamW's weight decay (1e-3).</summary>
    public double WeightDecay { get; set; } = 1e-3;
    public string Language { get; set; } = "en";
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    internal ConformerAsrConfiguration ToConfiguration()
    {
        int chunk = LookaheadFrames + 1;
        return new ConformerAsrConfiguration
        {
            SampleRate = SampleRate, NumMels = NumMels, EncoderDim = EncoderDim, NumEncoderLayers = NumEncoderLayers,
            NumAttentionHeads = NumAttentionHeads, FeedForwardMultiplier = FeedForwardMultiplier, ConvKernel = ConvKernel, RopeTheta = RopeTheta,
            Subsampling = ConformerAsrSubsampling.DepthwiseStriding8x, SubsamplingChannels = SubsamplingChannels, CausalDownsampling = true,
            FeatureNormalization = ConformerAsrFeatureNormalization.None, AttentionChunkSize = chunk, AttentionLeftChunks = LeftContextFrames / chunk,
            CausalConvolution = true, Decoder = ConformerAsrDecoder.Transducer, PredictionDim = PredictionDim, PredictionLayers = PredictionLayers,
            JointDim = JointDim, FastEmitLambda = FastEmitLambda, FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth,
            TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio, VocabSize = VocabSize,
        };
    }
}
