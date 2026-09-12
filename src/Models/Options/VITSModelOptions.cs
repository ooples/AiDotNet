using AiDotNet.Audio.TextToSpeech;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) models.
/// </summary>
public class VITSModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VITSModelOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public VITSModelOptions()
    {
        SpeakerEncoderPath = null;
        SampleRate = 22050;
        NumMels = 80;
        SpeakingRate = 1.0;
        NoiseScale = 0.667;
        LengthScale = 1.0;
        FftSize = 1024;
        HopLength = 256;
        HiddenDim = 192;
        NumHeads = 2;
        NumEncoderLayers = 6;
        NumFlowLayers = 4;
        SpeakerEmbeddingDim = 256;
        NumSpeakers = 1;
        MaxPhonemeLength = 256;
        PhonemeVocabSize = 128;
    }


    /// <summary>
    /// Gets or sets the speaker encoder path.
    /// </summary>
    public string? SpeakerEncoderPath { get; set; }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the num mels.
    /// </summary>
    public int NumMels { get; set; }

    /// <summary>
    /// Gets or sets the speaking rate.
    /// </summary>
    public double SpeakingRate { get; set; }

    /// <summary>
    /// Gets or sets the noise scale.
    /// </summary>
    public double NoiseScale { get; set; }

    /// <summary>
    /// Gets or sets the length scale.
    /// </summary>
    public double LengthScale { get; set; }

    /// <summary>
    /// Gets or sets the fft size.
    /// </summary>
    public int FftSize { get; set; }

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; }

    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the num encoder layers.
    /// </summary>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the num flow layers.
    /// </summary>
    public int NumFlowLayers { get; set; }

    /// <summary>
    /// Gets or sets the speaker embedding dim.
    /// </summary>
    public int SpeakerEmbeddingDim { get; set; }

    /// <summary>
    /// Gets or sets the num speakers.
    /// </summary>
    public int NumSpeakers { get; set; }

    /// <summary>
    /// Gets or sets the max phoneme length.
    /// </summary>
    public int MaxPhonemeLength { get; set; }

    /// <summary>
    /// Gets or sets the phoneme vocab size.
    /// </summary>
    public int PhonemeVocabSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
