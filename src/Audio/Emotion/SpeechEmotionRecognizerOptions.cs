using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// Configuration options for the SpeechEmotionRecognizer.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This model listens to a short clip of speech and says which emotion it
/// hears. It first turns the audio into a mel spectrogram — a picture of which pitches are loud
/// at each moment — and then runs a small image-style network over that picture. The settings
/// here describe both halves: how the picture is drawn, and how big the network is.
/// </para>
/// <para>
/// Defaults follow Badshah et al., "Speech Emotion Recognition from Spectrograms with Deep
/// Convolutional Neural Network" (IEEE PlatCon 2017), whose network is three convolutional
/// layers deep.
/// </para>
/// </remarks>
public class SpeechEmotionRecognizerOptions : ModelHyperparameterOptions
{
    /// <summary>The emotion set the model reports when the caller supplies none.</summary>
    /// <remarks>
    /// <para>
    /// Kept as the single source of the default so the two constructors cannot disagree about it.
    /// </para>
    /// </remarks>
    public static readonly string[] DefaultEmotionLabels =
    [
        "neutral",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgusted",
        "surprised"
    ];

    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public SpeechEmotionRecognizerOptions()
    {
        SampleRate = 16000;
        NumMels = 80;
        NFft = 1024;
        HopLength = 256;
        InputDurationSeconds = 3.0;
        NumConvBlocks = 3;
        BaseFilters = 32;
        HiddenDim = 256;
        DropoutRate = 0.3;
        LearningRate = 1e-4;
        IncludeArousalValence = true;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public SpeechEmotionRecognizerOptions(SpeechEmotionRecognizerOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        NumMels = other.NumMels;
        NFft = other.NFft;
        HopLength = other.HopLength;
        InputDurationSeconds = other.InputDurationSeconds;
        NumConvBlocks = other.NumConvBlocks;
        BaseFilters = other.BaseFilters;
        HiddenDim = other.HiddenDim;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        IncludeArousalValence = other.IncludeArousalValence;
        EmotionLabels = other.EmotionLabels;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the sample rate, in hertz, that input audio is expected to have. Default: 16000.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the number of mel frequency bands in the spectrogram. Default: 80.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many horizontal bands the spectrogram picture has. More
    /// bands means finer pitch detail and a taller picture for the network to process.</para>
    /// </remarks>
    public int NumMels { get; set; }

    /// <summary>
    /// Gets or sets the FFT window size, in samples. Default: 1024.
    /// </summary>
    public int NFft { get; set; }

    /// <summary>
    /// Gets or sets the stride between analysis frames, in samples. Default: 256.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far the analysis window slides between frames, which sets
    /// how wide the spectrogram picture is: a smaller hop gives more columns.</para>
    /// </remarks>
    public int HopLength { get; set; }

    /// <summary>
    /// Gets or sets the clip length, in seconds, the layers are built for. Default: 3.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Together with <see cref="SampleRate"/>, <see cref="NFft"/> and <see cref="HopLength"/> this
    /// fixes the frame count, which is the width of the input the convolutional stack expects.
    /// </para>
    /// </remarks>
    public double InputDurationSeconds { get; set; }

    /// <summary>
    /// Gets or sets the number of convolutional blocks. Default: 3, per Badshah et al.
    /// </summary>
    public int NumConvBlocks { get; set; }

    /// <summary>
    /// Gets or sets the number of filters in the first convolutional block. Default: 32.
    /// </summary>
    public int BaseFilters { get; set; }

    /// <summary>
    /// Gets or sets the width of the dense layer above the convolutional stack. Default: 256.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate applied in the dense head. Default: 0.3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Zero is a legitimate setting — it turns dropout off, and the model builds a shorter dense
    /// head to match — so this is deliberately not required positive.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; }

    /// <summary>
    /// Gets or sets the learning rate used when the model creates its own optimizer. Default: 1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Badshah et al. is an IEEE PlatCon publication whose optimizer settings are not available in
    /// an accessible source, so this is a documented library default rather than a paper value.
    /// Supplying an optimizer explicitly bypasses it entirely.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; }

    /// <summary>
    /// Gets or sets whether the model also reports arousal and valence alongside the emotion
    /// label. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Arousal is how energetic the speaker sounds and valence is how
    /// positive; together they place the emotion on a two-dimensional map rather than forcing it
    /// into one of the named categories.</para>
    /// </remarks>
    public bool IncludeArousalValence { get; set; }

    /// <summary>
    /// Gets or sets the emotion labels the model predicts over. Null uses
    /// <see cref="DefaultEmotionLabels"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The label count is the output width of the network, so changing this changes the model's
    /// shape and not merely how the results are named.
    /// </para>
    /// </remarks>
    public string[]? EmotionLabels { get; set; }

    /// <summary>
    /// Gets the emotion labels in effect, falling back to <see cref="DefaultEmotionLabels"/>.
    /// </summary>
    /// <returns>A non-empty label array.</returns>
    public string[] GetEffectiveEmotionLabels()
        => EmotionLabels is { Length: > 0 } labels ? labels : DefaultEmotionLabels;

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required value is zero or negative.</exception>
    /// <remarks>
    /// <para>
    /// <see cref="DropoutRate"/> is not checked: zero legitimately disables dropout.
    /// <see cref="IncludeArousalValence"/> is a switch, and <see cref="EmotionLabels"/> has a
    /// documented null fallback.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        Require(SampleRate, nameof(SampleRate));
        Require(NumMels, nameof(NumMels));
        Require(NFft, nameof(NFft));
        Require(HopLength, nameof(HopLength));
        Require(InputDurationSeconds, nameof(InputDurationSeconds));
        Require(NumConvBlocks, nameof(NumConvBlocks));
        Require(BaseFilters, nameof(BaseFilters));
        Require(HiddenDim, nameof(HiddenDim));
        Require(LearningRate, nameof(LearningRate));

        // Cross-field: the convolutional stack is built for a fixed frame count, and a clip too
        // short for one analysis window produces a zero- or negative-width input. Kept as a range
        // relationship rather than an unset-value check, so it throws ArgumentOutOfRangeException.
        int frames = (int)((InputDurationSeconds * SampleRate - NFft) / HopLength) + 1;
        if (frames < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(InputDurationSeconds),
                $"InputDurationSeconds {InputDurationSeconds} yields {frames} spectrogram frames at "
                    + $"SampleRate {SampleRate}, NFft {NFft} and HopLength {HopLength}; at least one is required.");
        }
    }
}
