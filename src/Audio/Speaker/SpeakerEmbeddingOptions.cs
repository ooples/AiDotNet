using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for speaker embedding extraction.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpeakerEmbedding model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeakerEmbeddingOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SpeakerEmbeddingOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SpeakerEmbeddingOptions(SpeakerEmbeddingOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        NumMfcc = other.NumMfcc;
        EmbeddingDimension = other.EmbeddingDimension;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of MFCC coefficients.
    /// </summary>
    public int NumMfcc { get; set; } = 40;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the path to the neural embedding model.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
