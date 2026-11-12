using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for Granite-speech: a conformer CTC encoder, a window Q-Former projector and a Granite LLM.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released granite-speech-3.3-8b: a 16-block conformer encoder of width
/// 1,024 (8 heads of 128, attention in blocks of 200 frames, kernel 15) over pairs of stacked 80-band log-mel frames; a
/// 2-layer Q-Former that reads each window of 15 encoder frames with 3 queries; and granite-3.3-8b-instruct (40 layers
/// of width 4,096, 32 query and 8 key-value heads, SwiGLU 12,800, a tied 49,160-row vocabulary) with LoRA of rank 64 on
/// the query and value projections. granite-speech-3.3-2b differs only in its LLM: width 2,048, SwiGLU 8,192, attention
/// multiplier 1/64 and logits scaling 8.</para>
/// </remarks>
public class GraniteSpeechOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GraniteSpeechOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GraniteSpeechOptions(GraniteSpeechOptions other)
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
        EncoderHeadDim = other.EncoderHeadDim;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        AttentionContextSize = other.AttentionContextSize;
        MaxRelativePosition = other.MaxRelativePosition;
        ConvKernel = other.ConvKernel;
        ConvExpansionFactor = other.ConvExpansionFactor;
        CtcVocabSize = other.CtcVocabSize;
        ProjectorDim = other.ProjectorDim;
        NumProjectorLayers = other.NumProjectorLayers;
        NumProjectorHeads = other.NumProjectorHeads;
        ProjectorFeedForwardDim = other.ProjectorFeedForwardDim;
        ProjectorWindowSize = other.ProjectorWindowSize;
        ProjectorDownsampleRate = other.ProjectorDownsampleRate;
        LlmDim = other.LlmDim;
        LlmFeedForwardDim = other.LlmFeedForwardDim;
        NumLlmLayers = other.NumLlmLayers;
        NumLlmAttentionHeads = other.NumLlmAttentionHeads;
        NumLlmKvHeads = other.NumLlmKvHeads;
        LlmRopeTheta = other.LlmRopeTheta;
        LlmRmsEpsilon = other.LlmRmsEpsilon;
        EmbeddingMultiplier = other.EmbeddingMultiplier;
        ResidualMultiplier = other.ResidualMultiplier;
        AttentionMultiplier = other.AttentionMultiplier;
        LogitsScaling = other.LogitsScaling;
        VocabSize = other.VocabSize;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        TrainingStage = other.TrainingStage;
        WarmupSteps = other.WarmupSteps;
        SystemPrompt = other.SystemPrompt;
        Instruction = other.Instruction;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>The longest recording transcribed (30 seconds).</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Log-mel bands (80); the encoder reads two stacked frames, 160 values.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>The encoder's width (1,024).</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>The encoder's conformer blocks (16 in the release; Table II of the paper lists 10).</summary>
    public int NumEncoderLayers { get; set; } = 16;
    /// <summary>The encoder's attention heads (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>The encoder's head width (128).</summary>
    public int EncoderHeadDim { get; set; } = 128;
    /// <summary>The encoder feed-forward width as a multiple of its width (4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;
    /// <summary>Encoder frames per attention block (200, 4 seconds).</summary>
    public int AttentionContextSize { get; set; } = 200;
    /// <summary>The largest relative position the Shaw embedding table holds (512).</summary>
    public int MaxRelativePosition { get; set; } = 512;
    /// <summary>The depthwise convolution kernel (15).</summary>
    public int ConvKernel { get; set; } = 15;
    /// <summary>The convolution module's inner width as a multiple of the encoder width (2).</summary>
    public int ConvExpansionFactor { get; set; } = 2;
    /// <summary>The CTC outputs, blank included (256 in the release; the paper's character set has 42).</summary>
    public int CtcVocabSize { get; set; } = 256;
    /// <summary>The Q-Former's width (1,024).</summary>
    public int ProjectorDim { get; set; } = 1024;
    /// <summary>The Q-Former's layers (2), each with self-attention, cross-attention and a feed-forward block.</summary>
    public int NumProjectorLayers { get; set; } = 2;
    /// <summary>The Q-Former's attention heads (16).</summary>
    public int NumProjectorHeads { get; set; } = 16;
    /// <summary>The Q-Former's feed-forward width (4,096).</summary>
    public int ProjectorFeedForwardDim { get; set; } = 4096;
    /// <summary>Encoder frames per Q-Former window, K (15).</summary>
    public int ProjectorWindowSize { get; set; } = 15;
    /// <summary>The projector's temporal downsampling K / N (5, so 3 queries per window).</summary>
    public int ProjectorDownsampleRate { get; set; } = 5;
    /// <summary>granite-3.3-8b's width (4,096).</summary>
    public int LlmDim { get; set; } = 4096;
    /// <summary>granite-3.3-8b's SwiGLU width (12,800).</summary>
    public int LlmFeedForwardDim { get; set; } = 12800;
    /// <summary>granite-3.3-8b's layers (40).</summary>
    public int NumLlmLayers { get; set; } = 40;
    /// <summary>granite-3.3-8b's query heads (32).</summary>
    public int NumLlmAttentionHeads { get; set; } = 32;
    /// <summary>granite-3.3-8b's key-value heads (8).</summary>
    public int NumLlmKvHeads { get; set; } = 8;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the language-model. Defaults to <c>10_000_000.0</c>.</summary>
    public double LlmRopeTheta { get; set; } = 10_000_000.0;
    /// <summary>Gets or sets the RMS-normalization stability constant for the language-model. Defaults to <c>1e-5</c>.</summary>
    public double LlmRmsEpsilon { get; set; } = 1e-5;
    /// <summary>Granite's scale on the input embeddings (12).</summary>
    public double EmbeddingMultiplier { get; set; } = 12.0;
    /// <summary>Granite's scale on each block's output before the residual sum (0.22).</summary>
    public double ResidualMultiplier { get; set; } = 0.22;
    /// <summary>Granite's attention score scale (1/128 for the 8B model, 1/64 for the 2B).</summary>
    public double AttentionMultiplier { get; set; } = 0.0078125;
    /// <summary>Granite's divisor of the logits (16 for the 8B model, 8 for the 2B).</summary>
    public double LogitsScaling { get; set; } = 16.0;
    /// <summary>The vocabulary rows (49,160). A vocabulary smaller than the tokenizer's has no chat prompt: the decoder
    /// reads the speech alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 49160;
    /// <summary>The LoRA rank on the query and value projections (64, the release's adapter).</summary>
    public int LoraRank { get; set; } = 64;
    /// <summary>Gets or sets the LoRA adapter scaling factor. Defaults to <c>32</c>.</summary>
    public double LoraAlpha { get; set; } = 32;
    /// <summary>Gets or sets the dropout probability of the LoRA adapter. Defaults to <c>0.0</c>.</summary>
    public double LoraDropout { get; set; } = 0.0;
    /// <summary>What training optimizes: every parameter on the transcript by default; the paper's stages are the
    /// encoder's CTC and then the projector and LoRA adapters.</summary>
    public GraniteSpeechTrainingStage TrainingStage { get; set; } = GraniteSpeechTrainingStage.Joint;
    /// <summary>Learning-rate warmup steps of the transcript stages' default optimizer (1,000, §4).</summary>
    public int WarmupSteps { get; set; } = 1000;
    /// <summary>The system turn (the model card's).</summary>
    public string SystemPrompt { get; set; } =
        "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant";
    /// <summary>The user's request after the audio (the model card's).</summary>
    public string Instruction { get; set; } = "can you transcribe the speech into a written format?";
    /// <summary>The longest transcript generated, in tokens.</summary>
    public int MaxTextLength { get; set; } = 200;
    /// <summary>Gets or sets the path to the model weights.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"en"</c>.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Granite's.</summary>
    public string[]? Vocabulary { get; set; }
}
