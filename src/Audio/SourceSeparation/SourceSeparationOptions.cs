using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Options for music source separation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SourceSeparation model. Default values follow the original paper settings.</para>
/// </remarks>
public class SourceSeparationOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SourceSeparationOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SourceSeparationOptions(SourceSeparationOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        StemCount = other.StemCount;
        HpssKernelSize = other.HpssKernelSize;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>Audio sample rate. Default: 44100.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>FFT size. Default: 4096.</summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>Hop length between frames. Default: 1024.</summary>
    public int HopLength { get; set; } = 1024;

    /// <summary>Number of stems to separate (2, 4, or 5). Default: 4.</summary>
    public int StemCount { get; set; } = 4;

    /// <summary>HPSS kernel size for spectral separation. Default: 31.</summary>
    public int HpssKernelSize { get; set; } = 31;

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
