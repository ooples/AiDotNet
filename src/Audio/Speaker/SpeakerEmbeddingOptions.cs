using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for speaker embedding extraction.
/// </summary>
public class SpeakerEmbeddingOptions : ModelOptions
{
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
