using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for Samba-ASR: Mamba-based state-space model for speech recognition.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SambaASR model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class SambaASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SambaASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SambaASROptions(SambaASROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        StateDimension = other.StateDimension;
        ExpandFactor = other.ExpandFactor;
        ConvKernelSize = other.ConvKernelSize;
        MaxSequenceLength = other.MaxSequenceLength;
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
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 24;
    /// <summary>Retained for backward compatibility; unused by the Mamba (attention-free) encoder.</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>SSM latent state size N per Mamba block (Gu &amp; Dao 2023). Default: 16.</summary>
    public int StateDimension { get; set; } = 16;
    /// <summary>Mamba inner expansion factor E (inner dim = E · EncoderDim). Default: 2.</summary>
    public int ExpandFactor { get; set; } = 2;
    /// <summary>Depthwise causal conv kernel width inside each Mamba block. Default: 4.</summary>
    public int ConvKernelSize { get; set; } = 4;
    /// <summary>Maximum encoder frame count the SSM scan is sized for. Default: 750.</summary>
    public int MaxSequenceLength { get; set; } = 750;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
