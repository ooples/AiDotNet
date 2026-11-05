using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for Phi-4-multimodal's speech path: a conformer audio encoder, a projector and Phi-4-Mini.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released Phi-4-multimodal-instruct: an audio encoder of three
/// convolutions (8× subsampling, an 80 ms token rate) and 24 conformer blocks of width 1,024 (16 heads, feed-forward
/// 1,536) over 80-band log filterbanks; a two-layer projector to 3,072; and Phi-4-Mini (32 layers of width 3,072, 24
/// query and 8 key-value heads, SwiGLU 8,192, a tied 200,064-row vocabulary, rotary embeddings on three quarters of each
/// head with LongRoPE) with the speech LoRA of rank 320 on every projection.</para>
/// </remarks>
public class Phi4AudioOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public Phi4AudioOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Phi4AudioOptions(Phi4AudioOptions other)
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
        EncoderFeedForwardDim = other.EncoderFeedForwardDim;
        SubsamplingChannels = other.SubsamplingChannels;
        ConvKernel = other.ConvKernel;
        RelativeBiasMaxDistance = other.RelativeBiasMaxDistance;
        LlmDim = other.LlmDim;
        LlmFeedForwardDim = other.LlmFeedForwardDim;
        NumLlmLayers = other.NumLlmLayers;
        NumLlmAttentionHeads = other.NumLlmAttentionHeads;
        NumLlmKvHeads = other.NumLlmKvHeads;
        LlmRopeTheta = other.LlmRopeTheta;
        PartialRotaryFactor = other.PartialRotaryFactor;
        UseLongRope = other.UseLongRope;
        LlmRmsEpsilon = other.LlmRmsEpsilon;
        VocabSize = other.VocabSize;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        TrainingStage = other.TrainingStage;
        Instruction = other.Instruction;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>The longest recording transcribed (30 seconds, the post-training limit for ASR).</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Log filterbank bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>The audio encoder's width (1,024).</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>The audio encoder's conformer blocks (24).</summary>
    public int NumEncoderLayers { get; set; } = 24;
    /// <summary>The audio encoder's attention heads (16).</summary>
    public int NumAttentionHeads { get; set; } = 16;
    /// <summary>The audio encoder's GLU feed-forward width (1,536).</summary>
    public int EncoderFeedForwardDim { get; set; } = 1536;
    /// <summary>Channels of the subsampling convolutions (1,024).</summary>
    public int SubsamplingChannels { get; set; } = 1024;
    /// <summary>The causal depthwise convolution kernel (3).</summary>
    public int ConvKernel { get; set; } = 3;
    /// <summary>The T5 relative attention bias's largest offset (500).</summary>
    public int RelativeBiasMaxDistance { get; set; } = 500;
    /// <summary>Phi-4-Mini's width (3,072).</summary>
    public int LlmDim { get; set; } = 3072;
    /// <summary>Phi-4-Mini's SwiGLU width (8,192).</summary>
    public int LlmFeedForwardDim { get; set; } = 8192;
    /// <summary>Phi-4-Mini's layers (32).</summary>
    public int NumLlmLayers { get; set; } = 32;
    /// <summary>Phi-4-Mini's query heads (24).</summary>
    public int NumLlmAttentionHeads { get; set; } = 24;
    /// <summary>Phi-4-Mini's key-value heads (8).</summary>
    public int NumLlmKvHeads { get; set; } = 8;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the language-model. Defaults to <c>10000.0</c>.</summary>
    public double LlmRopeTheta { get; set; } = 10000.0;
    /// <summary>The share of each head the rotary embedding turns (0.75).</summary>
    public double PartialRotaryFactor { get; set; } = 0.75;
    /// <summary>Whether the rotary embedding uses the release's LongRoPE factors (its 128K context extension); they
    /// apply to a head width of 128 with the default rotary share.</summary>
    public bool UseLongRope { get; set; } = true;
    /// <summary>Gets or sets the RMS-normalization stability constant for the language-model. Defaults to <c>1e-5</c>.</summary>
    public double LlmRmsEpsilon { get; set; } = 1e-5;
    /// <summary>The vocabulary rows (200,064). A vocabulary smaller than the tokenizer's has no chat prompt: the decoder
    /// reads the speech alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 200064;
    /// <summary>The speech LoRA's rank on every attention and feed-forward projection (320).</summary>
    public int LoraRank { get; set; } = 320;
    /// <summary>Gets or sets the LoRA adapter scaling factor. Defaults to <c>640</c>.</summary>
    public double LoraAlpha { get; set; } = 640;
    /// <summary>Gets or sets the dropout probability of the LoRA adapter. Defaults to <c>0.01</c>.</summary>
    public double LoraDropout { get; set; } = 0.01;
    /// <summary>What training updates: every parameter by default; the paper's stages are encoder alignment, then the
    /// projector and the speech LoRA.</summary>
    public Phi4AudioTrainingStage TrainingStage { get; set; } = Phi4AudioTrainingStage.Joint;
    /// <summary>The ASR instruction after the audio ("Transcribe the audio clip into text.", Appendix of the paper).</summary>
    public string Instruction { get; set; } = "Transcribe the audio clip into text.";
    /// <summary>The longest transcript generated, in tokens.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"en"</c>.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Phi's.</summary>
    public string[]? Vocabulary { get; set; }
}
