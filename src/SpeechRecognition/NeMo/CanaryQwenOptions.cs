using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Canary-Qwen: canary-1b-flash's Fast Conformer encoder, a linear projection and Qwen3-1.7B.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released nvidia/canary-qwen-2.5b: a Fast Conformer encoder of 32 blocks
/// at width 1,024 (8 heads, kernel 9, 8x depthwise-separable subsampling to 80 ms frames over 128 log-mel bands); a
/// linear projection to 2,048; and Qwen3-1.7B (28 layers of width 2,048, 16 query and 8 key-value heads of width 128,
/// SwiGLU 6,144, per-head query and key RMSNorm, a tied 151,936-row vocabulary) with LoRA of rank 128 (alpha 256,
/// dropout 0.01) on the query and value projections.</para>
/// </remarks>
public class CanaryQwenOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CanaryQwenOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CanaryQwenOptions(CanaryQwenOptions other)
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
        DecoderDim = other.DecoderDim;
        DecoderFeedForwardDim = other.DecoderFeedForwardDim;
        NumDecoderLayers = other.NumDecoderLayers;
        NumDecoderHeads = other.NumDecoderHeads;
        NumDecoderKvHeads = other.NumDecoderKvHeads;
        DecoderHeadDim = other.DecoderHeadDim;
        DecoderRopeTheta = other.DecoderRopeTheta;
        DecoderRmsEpsilon = other.DecoderRmsEpsilon;
        VocabSize = other.VocabSize;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        TrainingScope = other.TrainingScope;
        WarmupSteps = other.WarmupSteps;
        Instruction = other.Instruction;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    public int SampleRate { get; set; } = 16000;
    /// <summary>The longest recording transcribed (40 seconds, the longest in training).</summary>
    public int MaxAudioLengthSeconds { get; set; } = 40;
    /// <summary>Log-mel bands (128).</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>The encoder's width (1,024).</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>The encoder's Conformer blocks (32).</summary>
    public int NumEncoderLayers { get; set; } = 32;
    /// <summary>The encoder's attention heads (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>The encoder's feed-forward expansion (4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;
    /// <summary>The encoder's depthwise convolution kernel (9).</summary>
    public int ConvKernel { get; set; } = 9;
    /// <summary>Channels of the encoder's subsampling convolutions (256).</summary>
    public int SubsamplingChannels { get; set; } = 256;
    /// <summary>Qwen3-1.7B's width (2,048), the projection's output.</summary>
    public int DecoderDim { get; set; } = 2048;
    /// <summary>Qwen3-1.7B's SwiGLU width (6,144).</summary>
    public int DecoderFeedForwardDim { get; set; } = 6144;
    /// <summary>Qwen3-1.7B's layers (28).</summary>
    public int NumDecoderLayers { get; set; } = 28;
    /// <summary>Qwen3-1.7B's query heads (16).</summary>
    public int NumDecoderHeads { get; set; } = 16;
    /// <summary>Qwen3-1.7B's key-value heads (8).</summary>
    public int NumDecoderKvHeads { get; set; } = 8;
    /// <summary>Qwen3-1.7B's head width (128).</summary>
    public int DecoderHeadDim { get; set; } = 128;
    public double DecoderRopeTheta { get; set; } = 1000000.0;
    public double DecoderRmsEpsilon { get; set; } = 1e-6;
    /// <summary>The vocabulary rows (151,936). A vocabulary smaller than the tokenizer's has no chat prompt: the decoder
    /// reads the speech alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 151936;
    /// <summary>The LoRA rank on the query and value projections (128).</summary>
    public int LoraRank { get; set; } = 128;
    public double LoraAlpha { get; set; } = 256;
    public double LoraDropout { get; set; } = 0.01;
    /// <summary>What training updates: every parameter by default; the release freezes the LLM and trains the encoder,
    /// the projection and LoRA (<see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>).</summary>
    public SpeechLlmTrainingScope TrainingScope { get; set; } = SpeechLlmTrainingScope.AllParameters;
    /// <summary>The learning-rate warm-up steps (1,000).</summary>
    public int WarmupSteps { get; set; } = 1000;
    /// <summary>The ASR instruction before the audio ("Transcribe the following:", the model card's).</summary>
    public string Instruction { get; set; } = "Transcribe the following:";
    /// <summary>The longest transcript generated, in tokens.</summary>
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public string Language { get; set; } = "en";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Qwen's.</summary>
    public string[]? Vocabulary { get; set; }
}
