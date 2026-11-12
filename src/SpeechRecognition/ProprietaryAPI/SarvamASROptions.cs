using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>Options for the Sarvam speech-to-text stand-in, a Conformer encoder with a CTC layer over Indic-script characters.</summary>
/// <remarks>
/// <para>See <see cref="SarvamASR{T}"/> for which defaults the vendor or paper discloses and which the stand-in infers.</para>
/// <para><b>For Beginners:</b> These options set the size of the encoder and decoder, the feature front end and the training
/// augmentation.</para>
/// </remarks>
public class SarvamASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SarvamASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SarvamASROptions(SarvamASROptions other)
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

    /// <summary>Conformer width (512).</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Conformer blocks (18).</summary>
    public int NumEncoderLayers { get; set; } = 18;

    /// <summary>Attention heads per block (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Expansion of each half-step feed-forward module (Gulati et al. 2020, section 2.3: 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Depthwise convolution kernel (31).</summary>
    public int ConvKernel { get; set; } = 31;

    /// <summary>Rotary base standing in for the relative sinusoidal encoding.</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>SpecAugment frequency masks (Park et al. 2019; Gulati et al. 2020, section 3.1: 2).</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins (F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks (10).</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance (pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary size: token 0 is the blank and every other token is its Unicode code point, through the end of the Malayalam block (U+0D7F).</summary>
    public int VocabSize { get; set; } = 0x0D80;

    /// <summary>The text of each token; tokens past the list decode as their Unicode code points.</summary>
    public string[] Vocabulary { get; set; } = { "<blank>" };

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "hi";

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    internal ConformerAsrConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, EncoderDim = EncoderDim, NumEncoderLayers = NumEncoderLayers, NumAttentionHeads = NumAttentionHeads,
        FeedForwardMultiplier = FeedForwardMultiplier, ConvKernel = ConvKernel, RopeTheta = RopeTheta,
        FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio, VocabSize = VocabSize,
    };
}
