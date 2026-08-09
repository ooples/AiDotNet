using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.CTCVariants;

/// <summary>Options for CIF: Continuous Integrate-and-Fire mechanism for non-autoregressive ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CIFDecoder model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class CIFDecoderOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CIFDecoderOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CIFDecoderOptions(CIFDecoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        LearningRate = other.LearningRate;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>
    /// Gets or sets the AdamW learning rate used when no optimizer is injected. Default 2e-4.
    /// </summary>
    /// <remarks>
    /// This value was a literal <c>0.0002</c> inside the model's constructor, so it was not configurable
    /// at all: a caller who wanted a different rate had to build and inject an entire optimizer. The
    /// default is unchanged -- only its reachability is.
    /// </remarks>
    public double LearningRate { get; set; } = 2e-4;

    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}
