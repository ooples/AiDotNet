using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for Seed-ASR: the LUISE Conformer encoder, the frame-splicing converter and a decoder-only LLM.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> The paper fixes LUISE's depth (32 Conformer blocks), its size (nearly 2 billion parameters)
/// and its 40 ms frame rate over 80-band filterbanks, and the converter's four-frame splicing (160 ms per speech token).
/// It does not publish LUISE's width or the LLM (a mixture-of-experts model of over ten billion parameters) or its
/// vocabulary: the width defaults to 1,600, which gives nearly 2 billion parameters over 32 blocks, and the LLM to a dense
/// LLaMA-style decoder of 32 layers at width 4,096 with grouped-query attention.</para>
/// </remarks>
public class SeedASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SeedASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SeedASROptions(SeedASROptions other)
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
        SpliceFrames = other.SpliceFrames;
        LlmDim = other.LlmDim;
        LlmFeedForwardDim = other.LlmFeedForwardDim;
        NumLlmLayers = other.NumLlmLayers;
        NumLlmAttentionHeads = other.NumLlmAttentionHeads;
        NumLlmKvHeads = other.NumLlmKvHeads;
        LlmRopeTheta = other.LlmRopeTheta;
        LlmRmsEpsilon = other.LlmRmsEpsilon;
        VocabSize = other.VocabSize;
        TrainingScope = other.TrainingScope;
        Context = other.Context is null ? null : (int[])other.Context.Clone();
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Mel filterbank bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>LUISE's width; unpublished, 1,600 gives nearly 2 billion parameters over 32 blocks.</summary>
    public int EncoderDim { get; set; } = 1600;
    /// <summary>LUISE's Conformer blocks (32, §3.2).</summary>
    public int NumEncoderLayers { get; set; } = 32;
    /// <summary>LUISE's attention heads (unpublished; 16).</summary>
    public int NumAttentionHeads { get; set; } = 16;
    /// <summary>LUISE's feed-forward expansion (the Conformer's 4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;
    /// <summary>LUISE's depthwise convolution kernel (the Conformer's 31).</summary>
    public int ConvKernel { get; set; } = 31;
    /// <summary>Consecutive 40 ms frames the converter splices into one speech token (4, §3.3).</summary>
    public int SpliceFrames { get; set; } = 4;
    /// <summary>The LLM's width (unpublished; 4,096).</summary>
    public int LlmDim { get; set; } = 4096;
    /// <summary>The LLM's SwiGLU width (unpublished; 14,336).</summary>
    public int LlmFeedForwardDim { get; set; } = 14336;
    /// <summary>The LLM's layers (unpublished; 32).</summary>
    public int NumLlmLayers { get; set; } = 32;
    /// <summary>The LLM's query heads (unpublished; 32).</summary>
    public int NumLlmAttentionHeads { get; set; } = 32;
    /// <summary>The LLM's key-value heads (unpublished; 8).</summary>
    public int NumLlmKvHeads { get; set; } = 8;
    public double LlmRopeTheta { get; set; } = 500000.0;
    public double LlmRmsEpsilon { get; set; } = 1e-5;
    /// <summary>The LLM's vocabulary (unpublished; 32,000). Its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 32000;
    /// <summary>What training updates: every parameter by default; the paper's SFT trains the encoder and the converter
    /// with the LLM fixed (<see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>; Seed-ASR has no LoRA).</summary>
    public SpeechLlmTrainingScope TrainingScope { get; set; } = SpeechLlmTrainingScope.AllParameters;
    /// <summary>Context tokens placed before the speech (§3.4's context SFT); none for plain recognition.</summary>
    public int[]? Context { get; set; }
    /// <summary>The longest transcript generated, in tokens.</summary>
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public string Language { get; set; } = "en";
    /// <summary>Optional vocabulary mapping token ids to text; without one a token is its code point.</summary>
    public string[]? Vocabulary { get; set; }
    /// <summary>AdamW's learning rate (the paper states no optimizer).</summary>
    public double LearningRate { get; set; } = 1e-4;
    /// <summary>AdamW's weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
