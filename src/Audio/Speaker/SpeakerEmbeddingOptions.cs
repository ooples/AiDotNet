using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the SpeakerEmbeddingExtractor.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The embedding extractor turns a recording into a "voiceprint" — a short
/// list of numbers that identifies a speaker. The values here are the ones it ships with.
/// </para>
/// <para>
/// <b>#2090 note.</b> <see cref="FftSize"/>, <see cref="HopLength"/> and <see cref="NumMfcc"/>
/// were declared here before but never reached the MFCC front end, which always built itself
/// from its own defaults. They are now passed through, so changing them changes what the model
/// computes. The values are unchanged, so existing behaviour is unchanged.
/// </para>
/// </remarks>
public class SpeakerEmbeddingOptions : SpeakerRecognitionOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public SpeakerEmbeddingOptions()
    {
        SampleRate = 16000;
        EmbeddingDimension = 256;
        HiddenDim = 256;
        NumEncoderLayers = 3;
        FftSize = 512;
        HopLength = 160;
        NumMfcc = 40;
        MinimumDurationSeconds = 0.5;
        MaxFrames = 1000;
        DropoutRate = 0.0;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public SpeakerEmbeddingOptions(SpeakerEmbeddingOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        EmbeddingDimension = other.EmbeddingDimension;
        HiddenDim = other.HiddenDim;
        NumEncoderLayers = other.NumEncoderLayers;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        NumMfcc = other.NumMfcc;
        MinimumDurationSeconds = other.MinimumDurationSeconds;
        MaxFrames = other.MaxFrames;
        DropoutRate = other.DropoutRate;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the FFT window size, in samples, used by the MFCC front end. Default: 512.
    /// </summary>
    public int FftSize { get; set; }

    /// <summary>
    /// Gets or sets the stride between analysis frames, in samples. Default: 160 — 10 ms at the
    /// default 16 kHz sample rate, which is the usual frame rate for speech.
    /// </summary>
    public int HopLength { get; set; }

    /// <summary>
    /// Gets or sets the number of MFCC coefficients per frame. Default: 40.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many numbers summarise each short slice of audio. This is
    /// also the input width of the first layer, so the front end and the network are configured
    /// from the same value and cannot drift apart.</para>
    /// </remarks>
    public int NumMfcc { get; set; }

    /// <summary>
    /// Gets or sets the shortest recording, in seconds, the model will accept. Default: 0.5.
    /// </summary>
    public double MinimumDurationSeconds { get; set; }

    /// <summary>
    /// Gets or sets the longest input, in frames, the layers are built to handle. Default: 1000 —
    /// 10 seconds at the default hop length.
    /// </summary>
    public int MaxFrames { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate used in the encoder. Default: 0.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The x-vector network of Snyder et al. (2018) uses no dropout in its frame-level stack, so
    /// zero is the published setting rather than an unset value, and it is deliberately not
    /// required positive.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; }

    /// <summary>
    /// Gets or sets the path to an ONNX embedding model. Null selects the native trainable model.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options used when <see cref="ModelPath"/> is supplied.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required value is zero or negative.</exception>
    /// <remarks>
    /// <para>
    /// <see cref="DropoutRate"/> is not checked: zero is its published value.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        ValidateCore();
        Require(FftSize, nameof(FftSize));
        Require(HopLength, nameof(HopLength));
        Require(NumMfcc, nameof(NumMfcc));
        Require(MinimumDurationSeconds, nameof(MinimumDurationSeconds));
        Require(MaxFrames, nameof(MaxFrames));
    }
}
