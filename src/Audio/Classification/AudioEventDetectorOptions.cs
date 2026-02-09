using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for audio event detection.
/// </summary>
public class AudioEventDetectorOptions : ModelOptions
{
    /// <summary>Audio sample rate. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>FFT size. Default: 512.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Hop length. Default: 160.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Number of mel bands. Default: 64.</summary>
    public int NumMels { get; set; } = 64;

    /// <summary>Minimum frequency for mel filterbank. Default: 50.</summary>
    public int FMin { get; set; } = 50;

    /// <summary>Maximum frequency for mel filterbank. Default: 8000.</summary>
    public int FMax { get; set; } = 8000;

    /// <summary>Window size in seconds. Default: 1.0.</summary>
    public double WindowSize { get; set; } = 1.0;

    /// <summary>Window overlap ratio (0-1). Default: 0.5.</summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>Confidence threshold for event detection. Default: 0.3.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Custom event labels (optional).</summary>
    public string[]? CustomLabels { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
