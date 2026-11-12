using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for genre classification after Tzanetakis and Cook 2002: 23 ms analysis windows, 1 s texture windows,
/// 65536-sample beat histogram windows, and a three-component Gaussian mixture per genre.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GenreClassifier model. Default values follow the paper
/// (sections III and IV).</para>
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
        TextureWindowFrames = other.TextureWindowFrames;
        BeatWindow = other.BeatWindow;
        BeatHop = other.BeatHop;
        MixtureComponents = other.MixtureComponents;
        EmIterations = other.EmIterations;
        TopK = other.TopK;
        CustomGenres = other.CustomGenres;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>Audio sample rate. Default: 22050.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Analysis window (and FFT size) in samples. Default: 512, the paper's 23 ms at 22050 Hz.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Hop between analysis windows. Default: 512, non-overlapping windows.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Analysis windows per texture window. Default: 43, the paper's 1 s.</summary>
    public int TextureWindowFrames { get; set; } = 43;

    /// <summary>Samples per beat histogram window. Default: 65536 (about 3 s), as in the paper.</summary>
    public int BeatWindow { get; set; } = 65536;

    /// <summary>Hop between beat histogram windows. Default: 32768, half a window.</summary>
    public int BeatHop { get; set; } = 32768;

    /// <summary>Gaussians in each genre's mixture. Default: 3, the paper's GMM(3); 1 is its simple Gaussian classifier.</summary>
    public int MixtureComponents { get; set; } = 3;

    /// <summary>EM (and K-means initialization) iterations of a fit. Default: 100.</summary>
    public int EmIterations { get; set; } = 100;

    /// <summary>Number of top predictions to return. Default: 3.</summary>
    public int TopK { get; set; } = 3;

    /// <summary>Custom genre labels (optional).</summary>
    public string[]? CustomGenres { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
