using AiDotNet.Models.Options;
using AiDotNet.Onnx;
using AiDotNet.SpeechRecognition.Foundation;

namespace AiDotNet.SpeechRecognition.Multilingual;

/// <summary>Options for Chirp 2, built as Google USM-CTC: a BEST-RQ pre-trained Conformer with a CTC layer.</summary>
/// <remarks>
/// <para>The defaults are the paper's Conformer-2B (Zhang et al. 2023, Table 2: 32 layers, 1536 wide, 16 heads, kernel 5)
/// over 128-bin log-mel features (section 2.7), with BEST-RQ pre-training under a 16-codebook multi-softmax loss and
/// 8-second chunk-wise attention (section 2.4). EncoderDim 1024, NumEncoderLayers 24 and NumAttentionHeads 8 give
/// Conformer-0.6.</para>
/// <para><b>For Beginners:</b> These options set the size of Chirp 2's Conformer encoder, how it learns from
/// unlabeled speech, and how far its attention reaches.</para>
/// </remarks>
public class Chirp2Options : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public Chirp2Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Chirp2Options(Chirp2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        WindowLength = other.WindowLength;
        HopLength = other.HopLength;
        LogMelFloor = other.LogMelFloor;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        ConvKernel = other.ConvKernel;
        RopeTheta = other.RopeTheta;
        AttentionChunkSeconds = other.AttentionChunkSeconds;
        CodebookSize = other.CodebookSize;
        NumCodebooks = other.NumCodebooks;
        CodeDim = other.CodeDim;
        StackedFrames = other.StackedFrames;
        MaskProbability = other.MaskProbability;
        MaskLength = other.MaskLength;
        MaskNoiseStd = other.MaskNoiseStd;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
    }

    /// <summary>Audio sample rate in Hz (section 2.7: audio is resampled to 16 kHz).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio a transcription covers, in seconds; longer input is truncated.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Log-mel bins (section 2.7: 128-dimensional log-mel filterbank coefficients).</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Analysis window in samples (25 ms at 16 kHz; the paper does not state it).</summary>
    public int WindowLength { get; set; } = 400;

    /// <summary>Frame stride in samples (10 ms, BEST-RQ's, which the 4x reduction turns into 40 ms encoder frames).</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Floor added to the mel energies before the logarithm.</summary>
    public double LogMelFloor { get; set; } = 1e-6;

    /// <summary>Conformer width (Table 2, Conformer-2B: 1536; Conformer-0.6: 1024).</summary>
    public int EncoderDim { get; set; } = 1536;

    /// <summary>Conformer blocks (Table 2, Conformer-2B: 32; Conformer-0.6: 24).</summary>
    public int NumEncoderLayers { get; set; } = 32;

    /// <summary>Attention heads per block (Table 2, Conformer-2B: 16; Conformer-0.6: 8).</summary>
    public int NumAttentionHeads { get; set; } = 16;

    /// <summary>Feed-forward expansion of each half-step feed-forward module (Gulati et al. 2020: 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Depthwise convolution kernel (Table 2: 5).</summary>
    public int ConvKernel { get; set; } = 5;

    /// <summary>Rotary base of the attention's relative position encoding, which stands in for the paper's relative attention (section 2.1).</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>Chunk-wise attention window in seconds (section 2.4: an 8-second chunk was best); zero attends over the whole input.</summary>
    public int AttentionChunkSeconds { get; set; } = 8;

    /// <summary>Entries per random-projection codebook, which is also each pre-training softmax's width (BEST-RQ's default, which section 2.7 follows: 8192).</summary>
    public int CodebookSize { get; set; } = 8192;

    /// <summary>Codebooks of the multi-softmax loss, each with its own quantizer and softmax layer (sections 2.2.1 and 2.7: 16).</summary>
    public int NumCodebooks { get; set; } = 16;

    /// <summary>Codebook entry dimension, the height of each projection matrix (BEST-RQ's default: 16).</summary>
    public int CodeDim { get; set; } = 16;

    /// <summary>Input frames stacked per quantized label, matching the encoder's 4x reduction (BEST-RQ's default: 4).</summary>
    public int StackedFrames { get; set; } = 4;

    /// <summary>Probability that each input frame starts a masked span (BEST-RQ's default: 0.01).</summary>
    public double MaskProbability { get; set; } = 0.01;

    /// <summary>Masked span length in input frames (BEST-RQ's default: 400 ms, 40 frames at the 10 ms stride).</summary>
    public int MaskLength { get; set; } = 40;

    /// <summary>Standard deviation of the zero-mean normal noise that replaces masked frames (BEST-RQ's default: 0.1).</summary>
    public double MaskNoiseStd { get; set; } = 0.1;

    /// <summary>CTC vocabulary size, blank (token 0) included; tokens above 0 decode as Unicode code points. The paper tokenizes with word-piece models (section 2.7) without stating their size.</summary>
    public int VocabSize { get; set; } = 32000;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    // The network these options describe: BEST-RQ's, with rotary relative positions, attention within chunks of
    // AttentionChunkSeconds of 4x-reduced frames, and NumCodebooks quantizers.
    internal BestRqConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, WindowLength = WindowLength, HopLength = HopLength,
        LogMelFloor = LogMelFloor, EncoderDim = EncoderDim, NumEncoderLayers = NumEncoderLayers,
        NumAttentionHeads = NumAttentionHeads, FeedForwardMultiplier = FeedForwardMultiplier, ConvKernel = ConvKernel,
        RopeTheta = RopeTheta, AttentionChunkSize = AttentionChunkSeconds * SampleRate / (HopLength * StackedFrames),
        CodebookSize = CodebookSize, CodeDim = CodeDim, NumCodebooks = NumCodebooks, StackedFrames = StackedFrames,
        MaskProbability = MaskProbability, MaskLength = MaskLength, MaskNoiseStd = MaskNoiseStd, VocabSize = VocabSize,
    };
}
