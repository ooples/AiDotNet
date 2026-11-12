using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for SALM: a Fast Conformer audio encoder, a Conformer modality adapter and a Megatron GPT decoder.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the paper's: the 110M Fast Conformer-large encoder (17 blocks of width 512, 8
/// heads, kernel 9, 8x depthwise-separable subsampling to 80 ms frames over 80 log-mel bands); two Conformer layers of
/// width 512 with 4x subsampling as the modality adapter, so a speech token covers 320 ms; and the 2B Megatron GPT (24
/// layers of width 2,048 with 16 heads, SwiGLU 5,440, rotary embeddings on half of each head, LayerNorm, untied 256,000-row
/// embeddings) with LoRA of rank 128 on its fused query-key-value projection.</para>
/// </remarks>
public class SALMOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SALMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SALMOptions(SALMOptions other)
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
        AdapterDim = other.AdapterDim;
        NumAdapterLayers = other.NumAdapterLayers;
        NumAdapterHeads = other.NumAdapterHeads;
        LlmDim = other.LlmDim;
        LlmFeedForwardDim = other.LlmFeedForwardDim;
        NumLlmLayers = other.NumLlmLayers;
        NumLlmAttentionHeads = other.NumLlmAttentionHeads;
        LlmRopeTheta = other.LlmRopeTheta;
        PartialRotaryFactor = other.PartialRotaryFactor;
        LlmNormEpsilon = other.LlmNormEpsilon;
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

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Log-mel bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>The Fast Conformer encoder's width (512).</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>The encoder's Conformer blocks (17).</summary>
    public int NumEncoderLayers { get; set; } = 17;
    /// <summary>The encoder's attention heads (8).</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>The feed-forward expansion of the encoder and the adapter (4).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;
    /// <summary>The depthwise convolution kernel of the encoder and the adapter (9).</summary>
    public int ConvKernel { get; set; } = 9;
    /// <summary>Channels of the encoder's and the adapter's subsampling convolutions (256).</summary>
    public int SubsamplingChannels { get; set; } = 256;
    /// <summary>The modality adapter's width (512).</summary>
    public int AdapterDim { get; set; } = 512;
    /// <summary>The modality adapter's Conformer layers (2, §3.1).</summary>
    public int NumAdapterLayers { get; set; } = 2;
    /// <summary>The modality adapter's attention heads (8).</summary>
    public int NumAdapterHeads { get; set; } = 8;
    /// <summary>Megatron GPT's width (2,048).</summary>
    public int LlmDim { get; set; } = 2048;
    /// <summary>Megatron GPT's SwiGLU width (5,440).</summary>
    public int LlmFeedForwardDim { get; set; } = 5440;
    /// <summary>Megatron GPT's layers (24).</summary>
    public int NumLlmLayers { get; set; } = 24;
    /// <summary>Megatron GPT's attention heads (16).</summary>
    public int NumLlmAttentionHeads { get; set; } = 16;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the language-model. Defaults to <c>10000.0</c>.</summary>
    public double LlmRopeTheta { get; set; } = 10000.0;
    /// <summary>The share of each head the rotary embedding turns (<c>rotary_percentage</c> 0.5).</summary>
    public double PartialRotaryFactor { get; set; } = 0.5;
    /// <summary>Megatron GPT's LayerNorm epsilon.</summary>
    public double LlmNormEpsilon { get; set; } = 1e-5;
    /// <summary>The vocabulary rows (256,000). A vocabulary smaller than the tokenizer's has no prompt text: the decoder
    /// reads the speech alone and its last token ends the answer.</summary>
    public int VocabSize { get; set; } = 256000;
    /// <summary>The LoRA rank on the fused query-key-value projection (128, §3.1).</summary>
    public int LoraRank { get; set; } = 128;
    /// <summary>LoRA alpha; NeMo scales the update by alpha / rank with alpha equal to the rank.</summary>
    public double LoraAlpha { get; set; } = 128;
    /// <summary>Gets or sets the dropout probability of the LoRA adapter.</summary>
    public double LoraDropout { get; set; }
    /// <summary>What training updates: every parameter by default; the paper freezes the LLM and trains the encoder, the
    /// adapter and LoRA (<see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>).</summary>
    public SpeechLlmTrainingScope TrainingScope { get; set; } = SpeechLlmTrainingScope.AllParameters;
    /// <summary>The learning-rate warm-up steps (2,000, §4).</summary>
    public int WarmupSteps { get; set; } = 2000;
    /// <summary>The instruction after the speech. The paper's manifests are unpublished; this asks for an English
    /// transcript.</summary>
    public string Instruction { get; set; } = "Transcribe the audio into English text.";
    /// <summary>The longest transcript generated, in tokens (the release's <c>tokens_to_generate</c>, 128).</summary>
    public int MaxTextLength { get; set; } = 128;
    /// <summary>Gets or sets the path to the model weights.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"en"</c>.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Megatron's.</summary>
    public string[]? Vocabulary { get; set; }
}
