using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>Options for Moonshine Tiny.</summary>
/// <remarks>
/// <para>The defaults are Moonshine Tiny of Table 1: width 288, 6 encoder and 6 decoder layers, 8 heads.</para>
/// <para><b>For Beginners:</b> These options set the size of the encoder and decoder, the feature front end and
/// the training augmentation.</para>
/// </remarks>
public class MoonshineOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MoonshineOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MoonshineOptions(MoonshineOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        RopeTheta = other.RopeTheta;
        NumDecoderLayers = other.NumDecoderLayers;
        StartToken = other.StartToken;
        EndToken = other.EndToken;
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


    /// <summary>Conformer width.</summary>
    public int EncoderDim { get; set; } = 288;

    /// <summary>Conformer blocks.</summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>Attention heads per block.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Expansion of each feed-forward network (4x the width).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;


    /// <summary>Rotary base of every encoder and decoder self-attention (section 3.1: rotary position embeddings).</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>Decoder layers (Table 1).</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>The token that starts the decoder's input (Llama's beginning of sequence: 1).</summary>
    public int StartToken { get; set; } = 1;

    /// <summary>The token that ends a transcript (Llama's end of sequence: 2).</summary>
    public int EndToken { get; set; } = 2;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 194;

    /// <summary>SpecAugment frequency masks; Moonshine reads the waveform and trains without them.</summary>
    public int FrequencyMasks { get; set; } = 0;

    /// <summary>Largest frequency mask, in mel bins (section 3.1: F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks; none for Moonshine.</summary>
    public int TimeMasks { get; set; } = 0;

    /// <summary>Largest time mask as a fraction of the utterance (section 3.1: pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary (section 3.1: 32,000 BPE tokens plus 768 special tokens).</summary>
    public int VocabSize { get; set; } = 32768;

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
        SampleRate = SampleRate, EncoderDim = EncoderDim, NumEncoderLayers = NumEncoderLayers,
        NumAttentionHeads = NumAttentionHeads, FeedForwardMultiplier = FeedForwardMultiplier, RopeTheta = RopeTheta,
        Block = ConformerAsrBlock.Transformer, Decoder = ConformerAsrDecoder.Attention, Subsampling = ConformerAsrSubsampling.RawWaveformStem, EncoderFeedForward = ConformerAsrFeedForward.Gelu, DecoderFeedForward = ConformerAsrFeedForward.SwiGlu, DecoderRotary = true, AttentionDecoderLayers = NumDecoderLayers, AttentionDecoderFeedForward = 4 * EncoderDim, DecoderStartToken = StartToken, DecoderEndToken = EndToken, CtcWeight = 0, LabelSmoothing = 0, FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
        VocabSize = VocabSize,
    };
}
