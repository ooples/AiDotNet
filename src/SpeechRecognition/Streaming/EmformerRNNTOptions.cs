using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>Options for Emformer RNN-T.</summary>
/// <remarks>
/// <para>The defaults are torchaudio's <c>emformer_rnnt_base</c>: 20 Emformer layers of width 512 (128 x 4 stacked frames),
/// 8 heads, 2048 feed-forward units, 16-frame segments, a 4-frame look-ahead, 30 frames of left context, 1024-dimensional
/// encodings and a three-layer 512-unit LSTM prediction network.</para>
/// <para><b>For Beginners:</b> These options set the size of the streaming encoder, how far it looks back and ahead, and the
/// size of the decoder.</para>
/// </remarks>
public class EmformerRNNTOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EmformerRNNTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EmformerRNNTOptions(EmformerRNNTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        TimeReductionInputDim = other.TimeReductionInputDim;
        TimeReductionStride = other.TimeReductionStride;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        SegmentLength = other.SegmentLength;
        RightContextLength = other.RightContextLength;
        LeftContextLength = other.LeftContextLength;
        EncodingDim = other.EncodingDim;
        SymbolEmbeddingDim = other.SymbolEmbeddingDim;
        NumLstmLayers = other.NumLstmLayers;
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

    /// <summary>Log-mel bins (80).</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Width of the bias-free input linear layer before time reduction (128).</summary>
    public int TimeReductionInputDim { get; set; } = 128;

    /// <summary>Frames stacked by time reduction (4); the Emformer width is TimeReductionInputDim times this.</summary>
    public int TimeReductionStride { get; set; } = 4;

    /// <summary>Emformer layers (20).</summary>
    public int NumEncoderLayers { get; set; } = 20;

    /// <summary>Attention heads (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Feed-forward units of each Emformer layer (2048, GELU).</summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>Segment length in input frames (16).</summary>
    public int SegmentLength { get; set; } = 16;

    /// <summary>Look-ahead in input frames (4).</summary>
    public int RightContextLength { get; set; } = 4;

    /// <summary>Left context in reduced frames (30).</summary>
    public int LeftContextLength { get; set; } = 30;

    /// <summary>Width of the transcription and prediction encodings the joiner sums (1024).</summary>
    public int EncodingDim { get; set; } = 1024;

    /// <summary>Symbol embedding and LSTM width of the prediction network (512).</summary>
    public int SymbolEmbeddingDim { get; set; } = 512;

    /// <summary>LSTM layers of the prediction network (3).</summary>
    public int NumLstmLayers { get; set; } = 3;

    /// <summary>SpecAugment frequency masks.</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins.</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks.</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance.</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary, blank (token 0) included (torchaudio's LibriSpeech model: 4,097 symbols).</summary>
    public int VocabSize { get; set; } = 4097;

    /// <summary>The text of each token; tokens past the list decode as Unicode code points, and tokens in angle brackets
    /// decode to nothing.</summary>
    public string[] Vocabulary { get; set; } = { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    internal EmformerRnntConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, TimeReductionInputDim = TimeReductionInputDim, TimeReductionStride = TimeReductionStride,
        NumLayers = NumEncoderLayers, NumHeads = NumAttentionHeads, FeedForwardDim = FeedForwardDim, SegmentLength = SegmentLength,
        RightContextLength = RightContextLength, LeftContextLength = LeftContextLength, EncodingDim = EncodingDim,
        SymbolEmbeddingDim = SymbolEmbeddingDim, NumLstmLayers = NumLstmLayers, VocabSize = VocabSize, FrequencyMasks = FrequencyMasks,
        FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
    };
}
