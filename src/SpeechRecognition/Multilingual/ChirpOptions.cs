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

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
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
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 32000;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
}
