using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>Options for the streaming Zipformer transducer.</summary>
/// <remarks>
/// <para>The defaults are icefall's LibriSpeech Zipformer recipe (per-stack blocks 2, 2, 3, 4, 3, 2; widths 192, 256, 384,
/// 512, 384, 256; downsampling 1, 2, 4, 8, 4, 2; feed-forward 512, 768, 1024, 1536, 1024, 768; heads 4, 4, 4, 8, 4, 4;
/// kernels 31, 31, 15, 15, 15, 31), a 512-wide stateless decoder with two labels of context, a 512-wide joiner and a
/// 500-token vocabulary, with 32-frame chunks and 128 frames of left context for streaming.</para>
/// <para><b>For Beginners:</b> Each array has one entry per stack, from the input side to the output side.</para>
/// </remarks>
public class StreamingZipformerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public StreamingZipformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StreamingZipformerOptions(StreamingZipformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        NumLayersPerStack = (int[])other.NumLayersPerStack.Clone();
        Downsampling = (int[])other.Downsampling.Clone();
        FeedForwardDims = (int[])other.FeedForwardDims.Clone();
        NumHeads = (int[])other.NumHeads.Clone();
        EncoderDims = (int[])other.EncoderDims.Clone();
        ConvKernels = (int[])other.ConvKernels.Clone();
        QueryHeadDim = other.QueryHeadDim;
        ValueHeadDim = other.ValueHeadDim;
        PosHeadDim = other.PosHeadDim;
        PosDim = other.PosDim;
        EmbedChannels = (int[])other.EmbedChannels.Clone();
        DecoderDim = other.DecoderDim;
        JoinerDim = other.JoinerDim;
        ContextSize = other.ContextSize;
        ChunkSize = other.ChunkSize;
        LeftContextFrames = other.LeftContextFrames;
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

    /// <summary>Log-mel filterbank bins (80).</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Zipformer blocks of each stack.</summary>
    public int[] NumLayersPerStack { get; set; } = { 2, 2, 3, 4, 3, 2 };

    /// <summary>Downsampling factor of each stack relative to 50 Hz.</summary>
    public int[] Downsampling { get; set; } = { 1, 2, 4, 8, 4, 2 };

    /// <summary>Feed-forward width of each stack (the block's three modules use 3/4, 1 and 5/4 of it).</summary>
    public int[] FeedForwardDims { get; set; } = { 512, 768, 1024, 1536, 1024, 768 };

    /// <summary>Attention heads of each stack.</summary>
    public int[] NumHeads { get; set; } = { 4, 4, 4, 8, 4, 4 };

    /// <summary>Embedding width of each stack.</summary>
    public int[] EncoderDims { get; set; } = { 192, 256, 384, 512, 384, 256 };

    /// <summary>Depthwise convolution kernel of each stack.</summary>
    public int[] ConvKernels { get; set; } = { 31, 31, 15, 15, 15, 31 };

    /// <summary>Query and key dimension per head (32).</summary>
    public int QueryHeadDim { get; set; } = 32;

    /// <summary>Value dimension per head (12).</summary>
    public int ValueHeadDim { get; set; } = 12;

    /// <summary>Positional query dimension per head (4).</summary>
    public int PosHeadDim { get; set; } = 4;

    /// <summary>Width of the relative positional encoding (48).</summary>
    public int PosDim { get; set; } = 48;

    /// <summary>Channels of Conv-Embed's three convolutions (section 3.1: 8, 32, 128).</summary>
    public int[] EmbedChannels { get; set; } = { 8, 32, 128 };

    /// <summary>Width of the stateless decoder (512).</summary>
    public int DecoderDim { get; set; } = 512;

    /// <summary>Width of the joiner (512).</summary>
    public int JoinerDim { get; set; } = 512;

    /// <summary>Labels the stateless decoder reads (2).</summary>
    public int ContextSize { get; set; } = 2;

    /// <summary>Frames (at 50 Hz) of each streaming attention chunk; zero attends over the whole utterance.</summary>
    public int ChunkSize { get; set; } = 32;

    /// <summary>Frames (at 50 Hz) of left context each chunk attends to; negative for the whole past.</summary>
    public int LeftContextFrames { get; set; } = 128;

    /// <summary>SpecAugment frequency masks.</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins.</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks.</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance.</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary, blank (token 0) included (icefall's LibriSpeech BPE: 500).</summary>
    public int VocabSize { get; set; } = 500;

    /// <summary>The text of each token; tokens past the list decode as Unicode code points, and tokens in angle brackets
    /// decode to nothing.</summary>
    public string[] Vocabulary { get; set; } = { "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", " " };

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Total Zipformer blocks.</summary>
    public int NumEncoderLayers => NumLayersPerStack.Sum();

    internal ZipformerConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, NumLayers = NumLayersPerStack, Downsampling = Downsampling, FeedForwardDims = FeedForwardDims,
        NumHeads = NumHeads, EncoderDims = EncoderDims, ConvKernels = ConvKernels, QueryHeadDim = QueryHeadDim, ValueHeadDim = ValueHeadDim,
        PosHeadDim = PosHeadDim, PosDim = PosDim, EmbedChannels = EmbedChannels, DecoderDim = DecoderDim, JoinerDim = JoinerDim,
        ContextSize = ContextSize, VocabSize = VocabSize, ChunkSize = ChunkSize, LeftContextFrames = LeftContextFrames, Causal = ChunkSize > 0,
        FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio,
    };
}
