using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Localization;

/// <summary>
/// Options for sound source localization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SoundLocalizer model. Default values follow the original paper settings.</para>
/// </remarks>
public class SoundLocalizerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SoundLocalizerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SoundLocalizerOptions(SoundLocalizerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        SpeedOfSound = other.SpeedOfSound;
        Algorithm = other.Algorithm;
        AngleResolution = other.AngleResolution;
        FrameSize = other.FrameSize;
        CenterFrequency = other.CenterFrequency;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>Audio sample rate. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Speed of sound in m/s. Default: 343.</summary>
    public double SpeedOfSound { get; set; } = 343.0;

    /// <summary>Localization algorithm. Default: GCC-PHAT.</summary>
    public LocalizationAlgorithm Algorithm { get; set; } = LocalizationAlgorithm.GCCPHAT;

    /// <summary>Angular resolution in degrees. Default: 1.</summary>
    public double AngleResolution { get; set; } = 1.0;

    /// <summary>Frame size for MUSIC algorithm. Default: 512.</summary>
    public int FrameSize { get; set; } = 512;

    /// <summary>Center frequency for narrowband processing. Default: 1000 Hz.</summary>
    public double CenterFrequency { get; set; } = 1000.0;

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
