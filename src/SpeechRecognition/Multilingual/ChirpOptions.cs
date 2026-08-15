using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Multilingual;

/// <summary>Options for Chirp: Google's production multilingual ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Chirp model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class ChirpOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ChirpOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ChirpOptions(ChirpOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardExpansionFactor = other.FeedForwardExpansionFactor;
        ConvKernelSize = other.ConvKernelSize;
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
    public int EncoderDim { get; set; } = 1024;
    public int NumEncoderLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 16;

    /// <summary>
    /// Expansion factor of each Conformer block's macaron feed-forward modules: the FFN inner width is
    /// <c>EncoderDim × FeedForwardExpansionFactor</c>. Default 4, the Conformer paper's setting
    /// (Gulati et al. 2020, §2.1), which USM's Conformer encoder inherits.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each encoder block briefly widens the representation before
    /// projecting it back, giving the block extra capacity to transform features. 4× is the published
    /// setting; larger adds capacity and cost, smaller shrinks the model.</para>
    /// </remarks>
    public int FeedForwardExpansionFactor { get; set; } = 4;

    /// <summary>
    /// Kernel size of the depthwise convolution inside each Conformer block's convolution module.
    /// Default 5 — USM reports a 5-frame convolution kernel for its Conformer encoders
    /// (Zhang et al. 2023, arXiv:2303.01037 §2.1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many neighbouring audio frames the block's convolution
    /// looks at when capturing local sound patterns. Wider kernels see more context per block at
    /// higher cost.</para>
    /// </remarks>
    public int ConvKernelSize { get; set; } = 5;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 32000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
