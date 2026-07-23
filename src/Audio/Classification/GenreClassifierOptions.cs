using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for genre classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GenreClassifier model. Default values follow the original paper settings.</para>
/// </remarks>
public class GenreClassifierOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GenreClassifierOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GenreClassifierOptions(GenreClassifierOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        NumMfccs = other.NumMfccs;
        TopK = other.TopK;
        CustomGenres = other.CustomGenres;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>Audio sample rate. Default: 22050.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>FFT size. Default: 2048.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Hop length. Default: 512.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Number of MFCCs to extract. Default: 20.</summary>
    public int NumMfccs { get; set; } = 20;

    /// <summary>Number of top predictions to return. Default: 3.</summary>
    public int TopK { get; set; } = 3;

    /// <summary>Custom genre labels (optional).</summary>
    public string[]? CustomGenres { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
