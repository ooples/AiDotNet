using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for Fun-ASR-nano: an LLM-based recognizer of a SenseVoice-Small encoder, an adaptor and Qwen3-0.6B.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released Fun-ASR-Nano-2512 configuration: a 50+20-block SAN-M encoder of
/// width 512 over 80-band filterbanks stacked seven every six frames; a two-layer adaptor to Qwen3-0.6B's width of 1,024;
/// Qwen3-0.6B itself (28 layers, 16 query and 8 key-value heads of width 128, 151,936 tokens, tied embeddings); and a
/// five-layer CTC decoder over a 60,515-token vocabulary. Training uses AdamW at 5e-6 with 2,500 warmup steps and
/// gradient clipping at 5, as the release does.</para>
/// </remarks>
public class FunASRNanoOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public FunASRNanoOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FunASRNanoOptions(FunASRNanoOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        LfrM = other.LfrM;
        LfrN = other.LfrN;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumTpEncoderLayers = other.NumTpEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        EncoderFeedForwardDim = other.EncoderFeedForwardDim;
        FsmnKernelSize = other.FsmnKernelSize;
        AdaptorDownsampleRate = other.AdaptorDownsampleRate;
        AdaptorFeedForwardDim = other.AdaptorFeedForwardDim;
        NumAdaptorLayers = other.NumAdaptorLayers;
        AdaptorHeads = other.AdaptorHeads;
        LlmHiddenDim = other.LlmHiddenDim;
        NumLlmLayers = other.NumLlmLayers;
        LlmHeads = other.LlmHeads;
        LlmKeyValueHeads = other.LlmKeyValueHeads;
        LlmHeadDim = other.LlmHeadDim;
        LlmIntermediateDim = other.LlmIntermediateDim;
        LlmRmsEpsilon = other.LlmRmsEpsilon;
        LlmRopeTheta = other.LlmRopeTheta;
        VocabSize = other.VocabSize;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        CtcDecoderDim = other.CtcDecoderDim;
        CtcFeedForwardDim = other.CtcFeedForwardDim;
        NumCtcDecoderLayers = other.NumCtcDecoderLayers;
        CtcVocabSize = other.CtcVocabSize;
        MaxTextLength = other.MaxTextLength;
        SystemPrompt = other.SystemPrompt;
        Hotwords = other.Hotwords is null ? null : (string[])other.Hotwords.Clone();
        InverseTextNormalization = other.InverseTextNormalization;
        TrainingStage = other.TrainingStage;
        WarmupSteps = other.WarmupSteps;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    public int SampleRate { get; set; } = 16000;
    /// <summary>The longest recording transcribed (30 seconds).</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Filterbank bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Frames stacked per low-frame-rate step (7).</summary>
    public int LfrM { get; set; } = 7;
    /// <summary>Low-frame-rate stride in frames (6, a 60 ms step).</summary>
    public int LfrN { get; set; } = 6;
    /// <summary>The audio encoder's width (512).</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>The encoder's first group of SAN-M blocks (50).</summary>
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>The encoder's second group of SAN-M blocks (20).</summary>
    public int NumTpEncoderLayers { get; set; } = 20;
    /// <summary>The encoder's attention heads (4).</summary>
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>The encoder's feed-forward width (2,048).</summary>
    public int EncoderFeedForwardDim { get; set; } = 2048;
    /// <summary>The FSMN memory's kernel (11).</summary>
    public int FsmnKernelSize { get; set; } = 11;
    /// <summary>Encoder frames the adaptor concatenates per output (1).</summary>
    public int AdaptorDownsampleRate { get; set; } = 1;
    /// <summary>The adaptor's projection width (2,048).</summary>
    public int AdaptorFeedForwardDim { get; set; } = 2048;
    /// <summary>The adaptor's Transformer encoder layers (2).</summary>
    public int NumAdaptorLayers { get; set; } = 2;
    /// <summary>The attention heads of the adaptor's and the CTC decoder's Transformer layers (8).</summary>
    public int AdaptorHeads { get; set; } = 8;
    /// <summary>Qwen3-0.6B's width (1,024).</summary>
    public int LlmHiddenDim { get; set; } = 1024;
    /// <summary>Qwen3-0.6B's layers (28).</summary>
    public int NumLlmLayers { get; set; } = 28;
    /// <summary>Qwen3-0.6B's query heads (16).</summary>
    public int LlmHeads { get; set; } = 16;
    /// <summary>Qwen3-0.6B's key-value heads (8).</summary>
    public int LlmKeyValueHeads { get; set; } = 8;
    /// <summary>Qwen3-0.6B's head width (128).</summary>
    public int LlmHeadDim { get; set; } = 128;
    /// <summary>Qwen3-0.6B's SwiGLU width (3,072).</summary>
    public int LlmIntermediateDim { get; set; } = 3072;
    public double LlmRmsEpsilon { get; set; } = 1e-6;
    public double LlmRopeTheta { get; set; } = 1_000_000;
    /// <summary>The LLM's vocabulary (151,936, Qwen3's). A vocabulary smaller than the Qwen tokenizer's has no ChatML
    /// prompt: the LLM reads the speech alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 151936;
    /// <summary>The LoRA rank on the LLM's query and value projections (16, the release's <c>lora_conf</c>).</summary>
    public int LoraRank { get; set; } = 16;
    public double LoraAlpha { get; set; } = 32;
    public double LoraDropout { get; set; } = 0.05;
    /// <summary>The CTC decoder's width (512).</summary>
    public int CtcDecoderDim { get; set; } = 512;
    /// <summary>The CTC decoder's projection width (2,048).</summary>
    public int CtcFeedForwardDim { get; set; } = 2048;
    /// <summary>The CTC decoder's Transformer encoder layers (5).</summary>
    public int NumCtcDecoderLayers { get; set; } = 5;
    /// <summary>The CTC vocabulary (60,515); its last token is the blank.</summary>
    public int CtcVocabSize { get; set; } = 60515;
    /// <summary>The longest answer generated, in tokens.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>The ChatML system message ("You are a helpful assistant.").</summary>
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";
    /// <summary>Hotwords placed in the prompt's context section; none by default.</summary>
    public string[]? Hotwords { get; set; }
    /// <summary>Whether the transcript is inverse-text-normalized; false adds "，不进行文本规整" to the prompt.</summary>
    public bool InverseTextNormalization { get; set; } = true;
    /// <summary>The modules a training step updates: the report's §4.2 stages, which presuppose a pretrained Qwen3, or
    /// <see cref="FunASRNanoTrainingStage.Joint"/> (the default) for a model trained from its random initialization.</summary>
    public FunASRNanoTrainingStage TrainingStage { get; set; } = FunASRNanoTrainingStage.Joint;
    /// <summary>Linear warmup steps before the inverse-square-root decay of the release's AdamW recipe (2,500).</summary>
    public int WarmupSteps { get; set; } = 2500;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>The transcript's language named in the prompt; empty or "auto" leaves it unnamed.</summary>
    public string Language { get; set; } = "auto";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Qwen's.</summary>
    public string[]? Vocabulary { get; set; }
}
