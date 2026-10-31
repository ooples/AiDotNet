using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>Options for SPIRAL: self-supervised perturbation-invariant pre-training of a convolution-Transformer speech
/// encoder, with a convolutional CTC classifier.</summary>
/// <remarks>
/// <para>The defaults are the BASE model of Huang et al. 2022 (Table 1, section 4.2). For LARGE set FirstBlockLayers 4,
/// FirstBlockLayerDrop 0.05, EncoderDim 1024, NumEncoderLayers 20, NumAttentionHeads 16, FeedForwardDim 4096,
/// ProjectionDim 512, PredictorChannels 512, TeacherMomentumStart 0.990, TeacherMomentumEnd 0.999 and PretrainingSteps
/// 500000.</para>
/// <para><b>For Beginners:</b> These options set the size of SPIRAL's encoder, how its student is perturbed, and how its
/// teacher follows the student during pre-training.</para>
/// </remarks>
public class SPIRALOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SPIRALOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SPIRALOptions(SPIRALOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        WindowLength = other.WindowLength;
        HopLength = other.HopLength;
        LogMelFloor = other.LogMelFloor;
        FirstBlockDim = other.FirstBlockDim;
        FirstBlockLayers = other.FirstBlockLayers;
        FirstBlockHeads = other.FirstBlockHeads;
        FirstBlockFeedForwardDim = other.FirstBlockFeedForwardDim;
        FirstBlockLayerDrop = other.FirstBlockLayerDrop;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        LayerDrop = other.LayerDrop;
        DropoutRate = other.DropoutRate;
        ProjectionDim = other.ProjectionDim;
        PredictorChannels = other.PredictorChannels;
        ClassifierChannels = other.ClassifierChannels;
        Negatives = other.Negatives;
        Temperature = other.Temperature;
        TeacherMomentumStart = other.TeacherMomentumStart;
        TeacherMomentumEnd = other.TeacherMomentumEnd;
        PretrainingSteps = other.PretrainingSteps;
        MaxTeacherPaddingUnits = other.MaxTeacherPaddingUnits;
        TimeMaskProbability = other.TimeMaskProbability;
        TimeMaskLength = other.TimeMaskLength;
        FrequencyMaskProbability = other.FrequencyMaskProbability;
        FrequencyMaskLength = other.FrequencyMaskLength;
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

    /// <summary>Log-mel bins (section 4.2: 128-dimensional log-mel filterbank).</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Analysis window in samples (section 4.2: 20 ms).</summary>
    public int WindowLength { get; set; } = 320;

    /// <summary>Frame stride in samples (section 4.2: 10 ms).</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Floor added to the mel energies before the logarithm.</summary>
    public double LogMelFloor { get; set; } = 1e-6;

    /// <summary>Width of the first block's Transformer and its closing convolutions (Table 1: 512 for BASE and LARGE).</summary>
    public int FirstBlockDim { get; set; } = 512;

    /// <summary>Transformer layers in the first block (Table 1: BASE 2, LARGE 4).</summary>
    public int FirstBlockLayers { get; set; } = 2;

    /// <summary>Attention heads in the first block (Table 1: 8).</summary>
    public int FirstBlockHeads { get; set; } = 8;

    /// <summary>Feed-forward width in the first block (Table 1: 2048).</summary>
    public int FirstBlockFeedForwardDim { get; set; } = 2048;

    /// <summary>LayerDrop in the first block (Table 1: BASE 0, LARGE 0.05).</summary>
    public double FirstBlockLayerDrop { get; set; } = 0.0;

    /// <summary>Width of the second block, the encoder output (Table 1: BASE 768, LARGE 1024); its down-sampling convolution has twice this many channels.</summary>
    public int EncoderDim { get; set; } = 768;

    /// <summary>Transformer layers in the second block (Table 1: BASE 10, LARGE 20).</summary>
    public int NumEncoderLayers { get; set; } = 10;

    /// <summary>Attention heads in the second block (Table 1: BASE 12, LARGE 16).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Feed-forward width in the second block (Table 1: BASE 3072, LARGE 4096).</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>LayerDrop in the second block (Table 1: 0.05).</summary>
    public double LayerDrop { get; set; } = 0.05;

    /// <summary>Dropout in the convolutions and Transformers, for the student and the teacher alike (section 3.2; 0.1 in the released configuration).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Projection head output width (Table 1: BASE 256, LARGE 512).</summary>
    public int ProjectionDim { get; set; } = 256;

    /// <summary>Channels of the predictor's two convolutions (Table 1: BASE 256, LARGE 512).</summary>
    public int PredictorChannels { get; set; } = 256;

    /// <summary>Channels of the fine-tuning classifier's two kernel-5 convolutions (section 3.5: 512).</summary>
    public int ClassifierChannels { get; set; } = 512;

    /// <summary>Distractors per position in the in-utterance contrastive loss (100 in the released configuration; the paper does not state it).</summary>
    public int Negatives { get; set; } = 100;

    /// <summary>Contrastive temperature κ of Equation 2 (0.3 in the released configuration; the paper does not state it).</summary>
    public double Temperature { get; set; } = 0.3;

    /// <summary>Initial teacher moving-average rate α (section 4.2: BASE 0.995, LARGE 0.990).</summary>
    public double TeacherMomentumStart { get; set; } = 0.995;

    /// <summary>Final teacher moving-average rate α, reached by a cosine schedule (section 4.2: BASE 1.0, LARGE 0.999).</summary>
    public double TeacherMomentumEnd { get; set; } = 1.0;

    /// <summary>Pre-training updates the teacher's momentum schedule spans (section 4.2: BASE 200k, LARGE 500k).</summary>
    public int PretrainingSteps { get; set; } = 200000;

    /// <summary>Most padding added at each end of the teacher's input for position randomization, in units of the encoder's total stride (16 in the released configuration, capped at half the utterance).</summary>
    public int MaxTeacherPaddingUnits { get; set; } = 16;

    /// <summary>Proportion of time steps that start a SpecAugment time mask, filled with Gaussian noise (section 4.2: 0.025).</summary>
    public double TimeMaskProbability { get; set; } = 0.025;

    /// <summary>Length of each time mask in frames (section 4.2: 20).</summary>
    public int TimeMaskLength { get; set; } = 20;

    /// <summary>Proportion of mel bins that start a frequency mask, filled with zeros (section 4.2: 0.02).</summary>
    public double FrequencyMaskProbability { get; set; } = 0.02;

    /// <summary>Width of each frequency mask in bins (section 4.2: 20).</summary>
    public int FrequencyMaskLength { get; set; } = 20;

    /// <summary>CTC vocabulary size, blank (token 0) included; tokens above 0 decode as Unicode code points. Default 1024, the paper's SentencePiece subword count (section 3.5).</summary>
    public int VocabSize { get; set; } = 1024;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    // The network these options describe (Table 1): each block's down-sampling convolutions, the kernel-1 convolution
    // that sets its width, and its Transformer.
    internal SpiralConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, NumMels = NumMels, WindowLength = WindowLength, HopLength = HopLength, LogMelFloor = LogMelFloor,
        FirstConvolutions = new[] { (5, 384, 2), (5, FirstBlockDim, 2), (1, FirstBlockDim, 1) },
        FirstLayers = FirstBlockLayers, FirstHeads = FirstBlockHeads, FirstFeedForward = FirstBlockFeedForwardDim,
        FirstLayerDrop = FirstBlockLayerDrop,
        SecondConvolutions = new[] { (5, 2 * EncoderDim, 2), (1, EncoderDim, 1) },
        SecondLayers = NumEncoderLayers, SecondHeads = NumAttentionHeads, SecondFeedForward = FeedForwardDim, SecondLayerDrop = LayerDrop,
        Dropout = DropoutRate, ProjectionDim = ProjectionDim, PredictorChannels = PredictorChannels,
        ClassifierChannels = ClassifierChannels, VocabSize = VocabSize,
    };
}
