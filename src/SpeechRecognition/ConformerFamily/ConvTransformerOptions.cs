using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>Options for ConvTransformer.</summary>
/// <remarks>
/// <para>The defaults are the canonical model of section 3.1: width 1024, 10 encoder and 10 decoder Transformer blocks, 16 heads.</para>
/// <para><b>For Beginners:</b> These options set the size of the encoder and decoder, the feature front end and
/// the training augmentation.</para>
/// </remarks>
public class ConvTransformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ConvTransformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ConvTransformerOptions(ConvTransformerOptions other)
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
        DecoderEmbeddingDim = other.DecoderEmbeddingDim;
        NumMels = other.NumMels;
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


    /// <summary>Log-mel bins (section 3.1: 80-dimensional log mel-filterbank coefficients).</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Transformer width.</summary>
    public int EncoderDim { get; set; } = 1024;

    /// <summary>Conformer blocks.</summary>
    public int NumEncoderLayers { get; set; } = 10;

    /// <summary>Attention heads per block.</summary>
    public int NumAttentionHeads { get; set; } = 16;

    /// <summary>Expansion of each feed-forward network: 2048 ReLU units at width 1024.</summary>
    public int FeedForwardMultiplier { get; set; } = 2;


    /// <summary>Rotary base of the attention; zero, since the convolutional context replaces positional encodings (section 2.2).</summary>
    public double RopeTheta { get; set; } = 0.0;

    /// <summary>Decoder layers (Table 1).</summary>
    public int NumDecoderLayers { get; set; } = 10;

    /// <summary>Width of the decoder's token embedding (section 3.1: 512).</summary>
    public int DecoderEmbeddingDim { get; set; } = 512;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 194;

    /// <summary>SpecAugment frequency masks; Moonshine reads the waveform and trains without them.</summary>
    public int FrequencyMasks { get; set; } = 0;

    /// <summary>Largest frequency mask, in mel bins (section 3.1: F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks; none in the paper.</summary>
    public int TimeMasks { get; set; } = 0;

    /// <summary>Largest time mask as a fraction of the utterance (section 3.1: pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary (section 3.1: 5,000 unigram subword units).</summary>
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
        Block = ConformerAsrBlock.Transformer, Decoder = ConformerAsrDecoder.Attention, Subsampling = ConformerAsrSubsampling.VggBlocks, NoPositions = true, AttentionDecoderLayers = NumDecoderLayers, AttentionDecoderFeedForward = EncoderDim * FeedForwardMultiplier, DecoderContextConvolutions = 3, DecoderEmbeddingDim = DecoderEmbeddingDim, CtcWeight = 0, LabelSmoothing = 0, FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
        VocabSize = VocabSize,
    };
}
