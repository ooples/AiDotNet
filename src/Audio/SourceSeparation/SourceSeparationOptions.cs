using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Options for music source separation.
/// </summary>
public class SourceSeparationOptions : ModelOptions
{
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
