using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>Options for BEST-RQ: Conformer XL pre-trained with a random-projection quantizer, with a CTC layer.</summary>
/// <remarks>
/// <para>The defaults are the 0.6B non-streaming LibriSpeech model of Chiu et al. 2022, section 4.1: 80-bin log-mel input
/// at a 10 ms stride, Conformer XL (24 blocks, 1024 wide, 8 heads, kernel 5; Zhang et al. 2020, Table 1), an 8192-entry
/// 16-dimensional codebook over 4 stacked frames, and masking with probability 0.01 over 400 ms spans.</para>
/// <para><b>For Beginners:</b> These options configure the BEST-RQ recognizer: the size of its Conformer encoder, the
/// random quantizer that makes its self-supervised targets, and how much of the input pre-training hides.</para>
/// </remarks>
public class BESTRQOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public BESTRQOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BESTRQOptions(BESTRQOptions other)
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
        CodebookSize = other.CodebookSize;
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

    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio a transcription covers, in seconds; longer input is truncated.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Log-mel bins (section 4.1: 80-dimensional log-mel filter bank coefficients).</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Analysis window in samples (25 ms at 16 kHz; the paper states only the 10 ms stride).</summary>
    public int WindowLength { get; set; } = 400;

    /// <summary>Frame stride in samples (section 4.1: 10 ms).</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Floor added to the mel energies before the logarithm.</summary>
    public double LogMelFloor { get; set; } = 1e-6;

    /// <summary>Conformer width (Conformer XL, Zhang et al. 2020, Table 1: 1024).</summary>
    public int EncoderDim { get; set; } = 1024;

    /// <summary>Conformer blocks (section 4.1.1 and Conformer XL: 24).</summary>
    public int NumEncoderLayers { get; set; } = 24;

    /// <summary>Attention heads per block (Conformer XL: 8).</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Feed-forward expansion of each half-step feed-forward module (Gulati et al. 2020: 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Depthwise convolution kernel (Conformer XL: 5).</summary>
    public int ConvKernel { get; set; } = 5;

    /// <summary>Random-projection quantizer codebook entries, which is also the pre-training softmax width (section 4.1.1: 8192).</summary>
    public int CodebookSize { get; set; } = 8192;

    /// <summary>Codebook entry dimension, the height of the projection matrix (section 4.1.1: 16).</summary>
    public int CodeDim { get; set; } = 16;

    /// <summary>Input frames stacked per quantized label, matching the encoder's 4× reduction (section 4.1.1: 4).</summary>
    public int StackedFrames { get; set; } = 4;

    /// <summary>Probability that each input frame starts a masked span (section 4.1.1: 0.01).</summary>
    public double MaskProbability { get; set; } = 0.01;

    /// <summary>Masked span length in input frames (section 4.1.1: 400 ms, 40 frames at the 10 ms stride).</summary>
    public int MaskLength { get; set; } = 40;

    /// <summary>Standard deviation of the zero-mean normal noise that replaces masked frames (section 3: 0.1).</summary>
    public double MaskNoiseStd { get; set; } = 0.1;

    /// <summary>CTC vocabulary size, blank (token 0) included; tokens above 0 decode as Unicode code points. Default 1024, the size of the paper's fine-tuning WordPiece vocabulary (section 4.1).</summary>
    public int VocabSize { get; set; } = 1024;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";
}
