using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for audio event detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AudioEventDetector model. Default values follow the original paper settings.</para>
/// </remarks>
public class AudioEventDetectorOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public AudioEventDetectorOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AudioEventDetectorOptions(AudioEventDetectorOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        NumMels = other.NumMels;
        FMin = other.FMin;
        FMax = other.FMax;
        WindowSize = other.WindowSize;
        WindowOverlap = other.WindowOverlap;
        Threshold = other.Threshold;
        CustomLabels = other.CustomLabels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

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
