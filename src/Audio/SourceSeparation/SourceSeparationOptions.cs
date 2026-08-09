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

        // The native Demucs geometry. Omitting these made a clone silently revert to the paper
        // defaults while the original kept its configured (often much smaller) network -- the exact
        // silent-clone-data-loss this constructor exists to prevent, and what AIDN070 flags.
        DemucsDepth = other.DemucsDepth;
        DemucsBaseChannels = other.DemucsBaseChannels;
        DemucsKernelSize = other.DemucsKernelSize;
        DemucsStride = other.DemucsStride;
        DemucsPadding = other.DemucsPadding;
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

    /// <summary>Number of encoder levels in the native waveform-Demucs stack (mirrored in the decoder).</summary>
    /// <value>The encoder depth. Default 6, the paper's value.</value>
    /// <remarks>
    /// <para>
    /// Defossez et al. 2019 use six encoder levels. This was previously a <c>const int depth = 2</c>
    /// inside the model, with a comment saying the small value "keeps the invariant suite fast" -- a
    /// test constraint deciding a production model's capacity, and one a caller could not override.
    /// The paper's value is the default here; a test that needs a small fast network sets a small
    /// value, which is the direction that dependency should run.
    /// </para>
    /// <para>
    /// EACH LEVEL DIVIDES THE LENGTH BY <see cref="DemucsStride"/>, so the input must be at least
    /// <c>stride^depth</c> samples. At the defaults that is 4^6 = 4096 samples, about 0.09 s at 44.1 kHz.
    /// Shorter input with a deep stack is rejected rather than silently collapsed to a zero-length
    /// bottleneck.
    /// </para>
    /// <para><b>For Beginners:</b> How many times the network zooms out on the audio. More levels see
    /// longer-range structure and cost more time and memory. Lower it to 2 for a fast smoke test.</para>
    /// </remarks>
    public int DemucsDepth { get; set; } = 6;

    /// <summary>Channel count of the first Demucs encoder level; it doubles at each level.</summary>
    /// <value>The base channel count. Default 64, the paper's value.</value>
    /// <remarks>
    /// <para>
    /// Defossez et al. 2019 start at 64. The model previously hardcoded 8, which has no capacity to
    /// separate real music. At the default depth the top level is therefore 64 * 2^5 = 2048 channels.
    /// </para>
    /// <para><b>For Beginners:</b> How much the network can represent at each level. Larger separates
    /// better and costs more. Lower it to 8 alongside a small depth for a fast test.</para>
    /// </remarks>
    public int DemucsBaseChannels { get; set; } = 64;

    /// <summary>Convolution kernel size in the Demucs encoder and decoder.</summary>
    /// <value>The kernel size. Default 8, the paper's value.</value>
    public int DemucsKernelSize { get; set; } = 8;

    /// <summary>Convolution stride in the Demucs encoder and decoder.</summary>
    /// <value>The stride. Default 4, the paper's value.</value>
    /// <remarks>
    /// <para>Each encoder level divides the time axis by this, and each decoder level multiplies it
    /// back, which is what keeps the U-Net skip-add shapes aligned.</para>
    /// </remarks>
    public int DemucsStride { get; set; } = 4;

    /// <summary>Convolution padding in the Demucs encoder and decoder.</summary>
    /// <value>The padding. Default 2, which keeps encoder and decoder lengths aligned for the skip-add.</value>
    public int DemucsPadding { get; set; } = 2;
}
