using AiDotNet.Models.Options;
using AiDotNet.Onnx;
using AiDotNet.Enums;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for FireRedASR-LLM: LLM-enhanced version with Qwen2 decoder.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FireRedASRLLM model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
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

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        EncoderFeedForwardDim = other.EncoderFeedForwardDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        AdapterDim = other.AdapterDim;
        NumAdapterLayers = other.NumAdapterLayers;
        AdapterFrameSplicingFactor = other.AdapterFrameSplicingFactor;
        AdapterActivation = other.AdapterActivation;
        UseAdapterLayerNormalization = other.UseAdapterLayerNormalization;
        LlmDim = other.LlmDim;
        LlmFeedForwardDim = other.LlmFeedForwardDim;
        NumLlmLayers = other.NumLlmLayers;
        NumLlmAttentionHeads = other.NumLlmAttentionHeads;
        NumLlmKvHeads = other.NumLlmKvHeads;
        LlmRopeTheta = other.LlmRopeTheta;
        UseQwen2Decoder = other.UseQwen2Decoder;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the FireRedASR-AED-L encoder width.</summary>
    public int EncoderDim { get; set; } = 1280;
    /// <summary>Gets or sets the encoder feed-forward width.</summary>
    public int EncoderFeedForwardDim { get; set; } = 5120;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 16;
    /// <summary>Gets or sets the number of encoder attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 20;
    /// <summary>Gets or sets the adapter hidden/output width.</summary>
    public int AdapterDim { get; set; } = 3584;
    /// <summary>Gets or sets the number of nonlinear adapter projections before the output projection.</summary>
    public int NumAdapterLayers { get; set; } = 1;
    /// <summary>Gets or sets how many adjacent encoder frames are concatenated before the adapter.</summary>
    public int AdapterFrameSplicingFactor { get; set; } = 2;
    /// <summary>Gets or sets the adapter activation. The paper uses ReLU.</summary>
    public ActivationFunction AdapterActivation { get; set; } = ActivationFunction.ReLU;
    /// <summary>Gets or sets whether layer normalization is inserted between adapter projections.</summary>
    public bool UseAdapterLayerNormalization { get; set; } = false;
    /// <summary>Gets or sets the Qwen2-7B hidden width.</summary>
    public int LlmDim { get; set; } = 3584;
    /// <summary>Gets or sets the Qwen2-7B gated feed-forward width.</summary>
    public int LlmFeedForwardDim { get; set; } = 18944;
    /// <summary>Gets or sets the number of Qwen2-7B decoder layers.</summary>
    public int NumLlmLayers { get; set; } = 28;
    /// <summary>Gets or sets the number of Qwen2-7B attention heads.</summary>
    public int NumLlmAttentionHeads { get; set; } = 28;
    /// <summary>Gets or sets the number of Qwen2-7B key/value heads.</summary>
    public int NumLlmKvHeads { get; set; } = 4;
    /// <summary>Gets or sets the Qwen2 rotary-position base.</summary>
    public double LlmRopeTheta { get; set; } = 1_000_000.0;
    /// <summary>Gets or sets whether the decoder uses Qwen2 GQA, RoPE, RMSNorm, and gated SiLU blocks.</summary>
    public bool UseQwen2Decoder { get; set; } = true;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 152064;
    public int MaxTextLength { get; set; } = 32768;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
