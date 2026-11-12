using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for FireRedASR-LLM: the FireRedASR-AED encoder, an adapter and the Qwen2-7B-Instruct decoder.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released FireRedASR-LLM-L: FireRedASR-AED-L's Conformer encoder (16
/// blocks of width 1,280, 20 heads, kernel 33) over 80-band filterbanks, an adapter that splices two encoder frames into
/// one, and Qwen2-7B-Instruct (28 layers of width 3,584, 28 query and 4 key-value heads, SwiGLU 18,944, a 152,064-row
/// untied vocabulary) with LoRA of rank 64 on every projection.</para>
/// </remarks>
public class FireRedASRLLMOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public FireRedASRLLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FireRedASRLLMOptions(FireRedASRLLMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        EncoderDim = other.EncoderDim;
        EncoderFeedForwardDim = other.EncoderFeedForwardDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        ConvKernel = other.ConvKernel;
        RopeTheta = other.RopeTheta;
        AdapterFrameSplicingFactor = other.AdapterFrameSplicingFactor;
        LlmDim = other.LlmDim;
        LlmFeedForwardDim = other.LlmFeedForwardDim;
        NumLlmLayers = other.NumLlmLayers;
        NumLlmAttentionHeads = other.NumLlmAttentionHeads;
        NumLlmKvHeads = other.NumLlmKvHeads;
        LlmRopeTheta = other.LlmRopeTheta;
        LlmRmsEpsilon = other.LlmRmsEpsilon;
        VocabSize = other.VocabSize;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        TrainingScope = other.TrainingScope;
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
    /// <summary>Filterbank bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>The FireRedASR-AED-L encoder's width (1,280).</summary>
    public int EncoderDim { get; set; } = 1280;
    /// <summary>The encoder's feed-forward width (5,120, four times the width; must be a multiple of it).</summary>
    public int EncoderFeedForwardDim { get; set; } = 5120;
    /// <summary>The encoder's Conformer blocks (16).</summary>
    public int NumEncoderLayers { get; set; } = 16;
    /// <summary>The encoder's attention heads (20).</summary>
    public int NumAttentionHeads { get; set; } = 20;
    /// <summary>The encoder's depthwise convolution kernel (33).</summary>
    public int ConvKernel { get; set; } = 33;
    /// <summary>The rotary base standing in for the encoder's relative positional encoding, as in FireRedASR-AED.</summary>
    public double RopeTheta { get; set; } = 10000.0;
    /// <summary>Encoder frames the adapter concatenates into one (2).</summary>
    public int AdapterFrameSplicingFactor { get; set; } = 2;
    /// <summary>Qwen2-7B's width (3,584), which the adapter projects to.</summary>
    public int LlmDim { get; set; } = 3584;
    /// <summary>Qwen2-7B's SwiGLU width (18,944).</summary>
    public int LlmFeedForwardDim { get; set; } = 18944;
    /// <summary>Qwen2-7B's layers (28).</summary>
    public int NumLlmLayers { get; set; } = 28;
    /// <summary>Qwen2-7B's query heads (28).</summary>
    public int NumLlmAttentionHeads { get; set; } = 28;
    /// <summary>Qwen2-7B's key-value heads (4).</summary>
    public int NumLlmKvHeads { get; set; } = 4;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the language-model. Defaults to <c>1_000_000.0</c>.</summary>
    public double LlmRopeTheta { get; set; } = 1_000_000.0;
    /// <summary>Gets or sets the RMS-normalization stability constant for the language-model. Defaults to <c>1e-6</c>.</summary>
    public double LlmRmsEpsilon { get; set; } = 1e-6;
    /// <summary>Qwen2-7B's vocabulary rows (152,064). A vocabulary smaller than the tokenizer's has no ChatML prompt: the
    /// decoder reads the speech alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 152064;
    /// <summary>The LoRA rank on every attention and feed-forward projection (64, the release's <c>LoraConfig</c>).</summary>
    public int LoraRank { get; set; } = 64;
    /// <summary>Gets or sets the LoRA adapter scaling factor. Defaults to <c>16</c>.</summary>
    public double LoraAlpha { get; set; } = 16;
    /// <summary>Gets or sets the dropout probability of the LoRA adapter. Defaults to <c>0.05</c>.</summary>
    public double LoraDropout { get; set; } = 0.05;
    /// <summary>The parameters training updates: every one by default; the paper's recipe trains the encoder, the adapter
    /// and the LoRA adapters only.</summary>
    public SpeechLlmTrainingScope TrainingScope { get; set; } = SpeechLlmTrainingScope.AllParameters;
    /// <summary>The instruction after the speech ("请转写音频为文字", transcribe the audio as text).</summary>
    public string Instruction { get; set; } = "请转写音频为文字";
    /// <summary>The longest transcript generated, in tokens; the release also stops at the adapter's output length.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path to the model weights.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"zh"</c>.</summary>
    public string Language { get; set; } = "zh";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Qwen's.</summary>
    public string[]? Vocabulary { get; set; }
}
