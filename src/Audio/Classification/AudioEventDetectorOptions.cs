using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for audio event detection with Hershey et al. 2017's audio ResNet-50: 96 x 64 log-mel patches of 960 ms
/// (25 ms windows every 10 ms, 64 mel bins), a ResNet-50 whose first convolution has no stride, and a sigmoid output.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AudioEventDetector model. Default values follow the paper
/// (section 3).</para>
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
        PatchFrames = other.PatchFrames;
        BaseChannels = other.BaseChannels;
        StageDepths = (int[])other.StageDepths.Clone();
    }

    /// <summary>Audio sample rate. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>STFT window in samples. Default: 400, the paper's 25 ms at 16 kHz.</summary>
    public int FftSize { get; set; } = 400;

    /// <summary>Hop length. Default: 160, every 10 ms.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Number of mel bands. Default: 64.</summary>
    public int NumMels { get; set; } = 64;

    /// <summary>Minimum frequency for mel filterbank. Default: 125, the band edge of the authors' released AudioSet
    /// front end (the paper does not state it).</summary>
    public int FMin { get; set; } = 125;

    /// <summary>Maximum frequency for mel filterbank. Default: 7500, the released front end's upper edge.</summary>
    public int FMax { get; set; } = 7500;

    /// <summary>Window size in seconds. Default: 0.96 (section 3.1: "non-overlapping 960 ms frames").</summary>
    public double WindowSize { get; set; } = 0.96;

    /// <summary>Window overlap ratio (0-1). Default: 0, non-overlapping.</summary>
    public double WindowOverlap { get; set; } = 0.0;

    /// <summary>Spectrogram frames per classifier input. Default: 96, giving the paper's 96 x 64 patches.</summary>
    public int PatchFrames { get; set; } = 96;

    /// <summary>Channels of the first convolution and of the first bottleneck stage; each later stage doubles them and
    /// a bottleneck expands its stage width four times. Default: 64 (ResNet-50).</summary>
    public int BaseChannels { get; set; } = 64;

    /// <summary>Bottleneck blocks per stage. Default: { 3, 4, 6, 3 } (ResNet-50).</summary>
    public int[] StageDepths { get; set; } = { 3, 4, 6, 3 };

    /// <summary>Confidence threshold for event detection. Default: 0.3.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Custom event labels (optional).</summary>
    public string[]? CustomLabels { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
