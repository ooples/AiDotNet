using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>Options for the Deepgram Nova-2 stand-in, a Transformer encoder with a CTC layer (Deepgram has disclosed only a Transformer-based end-to-end model).</summary>
/// <remarks>
/// <para>See <see cref="DeepgramNova2{T}"/> for which defaults the vendor or paper discloses and which the stand-in infers.</para>
/// <para><b>For Beginners:</b> These options set the size of the encoder and decoder, the feature front end and the training
/// augmentation.</para>
/// </remarks>
public class DeepgramNova2Options : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DeepgramNova2Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DeepgramNova2Options(DeepgramNova2Options other)
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

    /// <summary>Transformer width (the base Transformer: 512).</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Transformer encoder layers (12).</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Attention heads per layer (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Feed-forward expansion (2,048 units at width 512).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>SpecAugment frequency masks (Park et al. 2019; Gulati et al. 2020, section 3.1: 2).</summary>
    public int FrequencyMasks { get; set; } = 2;

    /// <summary>Largest frequency mask, in mel bins (F = 27).</summary>
    public int FrequencyMaskWidth { get; set; } = 27;

    /// <summary>SpecAugment time masks (10).</summary>
    public int TimeMasks { get; set; } = 10;

    /// <summary>Largest time mask as a fraction of the utterance (pS = 0.05).</summary>
    public double TimeMaskRatio { get; set; } = 0.05;

    /// <summary>Output vocabulary size, blank (token 0) included.</summary>
    public int VocabSize { get; set; } = 34;

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
        Block = ConformerAsrBlock.Transformer, RopeTheta = 0, FeedForwardMultiplier = FeedForwardMultiplier,
        FrequencyMasks = FrequencyMasks, FrequencyMaskWidth = FrequencyMaskWidth, TimeMasks = TimeMasks, TimeMaskRatio = TimeMaskRatio, VocabSize = VocabSize,
    };
}
