using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for Qwen3-ASR-1.7B: an AuT audio encoder, a projector and the Qwen3-1.7B decoder (Qwen3-ASR technical report, 2026).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released Qwen3-ASR-1.7B configuration: 128-band Whisper log-mel features, an
/// AuT encoder of 24 layers at width 1024 (16 heads, feed-forward 4096) that emits 12.5 tokens per second,
/// and Qwen3-1.7B (2,048 wide, 28 layers, 16 query and 8 key-value heads of width 128, SwiGLU 6,144, 151,936 tokens, tied
/// embeddings). Fine-tuning uses AdamW at 2e-5 with a linear schedule, as the release's SFT script does.</para>
/// </remarks>
public class Qwen3ASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public Qwen3ASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Qwen3ASROptions(Qwen3ASROptions other)
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
        DownsampleChannels = other.DownsampleChannels;
        ChunkFrames = other.ChunkFrames;
        WindowChunks = other.WindowChunks;
        LlmHiddenDim = other.LlmHiddenDim;
        NumLlmLayers = other.NumLlmLayers;
        LlmHeads = other.LlmHeads;
        LlmKeyValueHeads = other.LlmKeyValueHeads;
        LlmHeadDim = other.LlmHeadDim;
        LlmIntermediateDim = other.LlmIntermediateDim;
        LlmRmsEpsilon = other.LlmRmsEpsilon;
        LlmRopeTheta = other.LlmRopeTheta;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        Context = other.Context;
        WarmupSteps = other.WarmupSteps;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>The longest recording transcribed (20 minutes, the report's single-utterance limit).</summary>
    public int MaxAudioLengthSeconds { get; set; } = 1200;
    /// <summary>Whisper log-mel bands (128).</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>The AuT encoder's width (1024).</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>The AuT encoder's layers (24).</summary>
    public int NumEncoderLayers { get; set; } = 24;
    /// <summary>The AuT encoder's attention heads (16).</summary>
    public int NumAttentionHeads { get; set; } = 16;
    /// <summary>The AuT encoder's feed-forward width (4096).</summary>
    public int EncoderFeedForwardDim { get; set; } = 4096;
    /// <summary>The convolutional front end's channels (480).</summary>
    public int DownsampleChannels { get; set; } = 480;
    /// <summary>Spectrogram frames per convolution chunk (100, one second).</summary>
    public int ChunkFrames { get; set; } = 100;
    /// <summary>Chunks per encoder attention window (8).</summary>
    public int WindowChunks { get; set; } = 8;
    /// <summary>Qwen3-1.7B's width (2,048).</summary>
    public int LlmHiddenDim { get; set; } = 2048;
    /// <summary>Qwen3-1.7B's layers (28).</summary>
    public int NumLlmLayers { get; set; } = 28;
    /// <summary>Qwen3-1.7B's query heads (16).</summary>
    public int LlmHeads { get; set; } = 16;
    /// <summary>Qwen3-1.7B's key-value heads (8).</summary>
    public int LlmKeyValueHeads { get; set; } = 8;
    /// <summary>Qwen3-1.7B's head width (128).</summary>
    public int LlmHeadDim { get; set; } = 128;
    /// <summary>Qwen3-1.7B's SwiGLU width (6,144).</summary>
    public int LlmIntermediateDim { get; set; } = 6144;
    /// <summary>Gets or sets the RMS-normalization stability constant for the language-model. Defaults to <c>1e-6</c>.</summary>
    public double LlmRmsEpsilon { get; set; } = 1e-6;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the language-model. Defaults to <c>1_000_000</c>.</summary>
    public double LlmRopeTheta { get; set; } = 1_000_000;
    /// <summary>The decoder's vocabulary (151,936). A vocabulary smaller than the Qwen3-ASR tokenizer's has no ChatML
    /// prompt: the decoder reads the audio alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 151936;
    /// <summary>The longest answer generated, in tokens.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Context text (hotwords, background) placed in the system message; empty by default.</summary>
    public string Context { get; set; } = string.Empty;
    /// <summary>Linear warmup steps. The release's SFT script warms up over 2% of the run, whose length is not known when
    /// the model is built; 0 starts at the peak rate.</summary>
    public int WarmupSteps { get; set; }
    /// <summary>Gets or sets the path to the model weights.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>The language to force ("English", "Chinese", …): the prompt then ends with <c>language {Language}&lt;asr_text&gt;</c>
    /// and the answer is the transcript alone. "auto" lets the model name the language first.</summary>
    public string Language { get; set; } = "auto";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Qwen's.</summary>
    public string[]? Vocabulary { get; set; }
}
