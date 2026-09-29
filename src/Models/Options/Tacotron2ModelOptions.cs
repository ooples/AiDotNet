using AiDotNet.Audio.TextToSpeech;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Tacotron 2 text-to-speech models.
/// </summary>
public class Tacotron2ModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Tacotron2ModelOptions"/> class carrying
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
    public Tacotron2ModelOptions()
    {
        VocoderPath = null;
        SampleRate = 22050;
        NumMels = 80;
        SpeakingRate = 1.0;
        MaxDecoderSteps = 1000;
        StopThreshold = 0.5;
        FftSize = 1024;
        HopLength = 256;
        GriffinLimIterations = 60;
        VocabSize = 148;
        EmbeddingDim = 512;
        EncoderDim = 512;
        DecoderDim = 1024;
        AttentionDim = 128;
        AttentionFilters = 32;
        PrenetDim = 256;
        PostnetEmbeddingDim = 512;
        NumEncoderConvLayers = 3;
        NumPostnetConvLayers = 5;
        NumMelsPerFrame = 2;
    }


    /// <summary>
    /// Gets or sets the vocoder path.
    /// </summary>
    public string? VocoderPath { get; set; }

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
    /// Gets or sets the max decoder steps.
    /// </summary>
    public int MaxDecoderSteps { get; set; }

    /// <summary>
    /// Gets or sets the stop threshold.
    /// </summary>
    public double StopThreshold { get; set; }

    /// <summary>
    /// Gets or sets the fft size.
    /// </summary>
    public int FftSize { get; set; }

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; }

    /// <summary>
    /// Gets or sets the griffin lim iterations.
    /// </summary>
    public int GriffinLimIterations { get; set; }

    /// <summary>
    /// Gets or sets the vocab size.
    /// </summary>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the embedding dim.
    /// </summary>
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Gets or sets the encoder dim.
    /// </summary>
    public int EncoderDim { get; set; }

    /// <summary>
    /// Gets or sets the decoder dim.
    /// </summary>
    public int DecoderDim { get; set; }

    /// <summary>
    /// Gets or sets the attention dim.
    /// </summary>
    public int AttentionDim { get; set; }

    /// <summary>
    /// Gets or sets the attention filters.
    /// </summary>
    public int AttentionFilters { get; set; }

    /// <summary>
    /// Gets or sets the prenet dim.
    /// </summary>
    public int PrenetDim { get; set; }

    /// <summary>
    /// Gets or sets the postnet embedding dim.
    /// </summary>
    public int PostnetEmbeddingDim { get; set; }

    /// <summary>
    /// Gets or sets the num encoder conv layers.
    /// </summary>
    public int NumEncoderConvLayers { get; set; }

    /// <summary>
    /// Gets or sets the num postnet conv layers.
    /// </summary>
    public int NumPostnetConvLayers { get; set; }

    /// <summary>
    /// Gets or sets the num mels per frame.
    /// </summary>
    public int NumMelsPerFrame { get; set; }

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
