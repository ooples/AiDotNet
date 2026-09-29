using AiDotNet.Audio.SpeechRecognition;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Wav2Vec2 speech recognition models.
/// </summary>
public class Wav2Vec2ModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Wav2Vec2ModelOptions"/> class carrying
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
    public Wav2Vec2ModelOptions()
    {
        Language = "en";
        SampleRate = 16000;
        MaxAudioLengthSeconds = 30;
        HiddenDim = 768;
        NumTransformerLayers = 12;
        NumHeads = 12;
        FfDim = 3072;
    }


    /// <summary>
    /// Gets or sets the language.
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the max audio length seconds.
    /// </summary>
    public int MaxAudioLengthSeconds { get; set; }

    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num transformer layers.
    /// </summary>
    public int NumTransformerLayers { get; set; }

    /// <summary>
    /// Gets or sets the num heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the ff dim.
    /// </summary>
    public int FfDim { get; set; }

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
