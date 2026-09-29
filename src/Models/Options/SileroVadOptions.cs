using AiDotNet.Audio.VoiceActivity;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Silero VAD (Voice Activity Detection) models.
/// </summary>
public class SileroVadOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SileroVadOptions"/> class carrying
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
    public SileroVadOptions()
    {
        SampleRate = 16000;
        FrameSize = 512;
        Threshold = 0.5;
        MinSpeechDurationMs = 250;
        MinSilenceDurationMs = 100;
        ConvFilters = 64;
        LstmHiddenDim = 64;
        NumLstmLayers = 2;
    }


    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the frame size.
    /// </summary>
    public int FrameSize { get; set; }

    /// <summary>
    /// Gets or sets the threshold.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets the min speech duration ms.
    /// </summary>
    public int MinSpeechDurationMs { get; set; }

    /// <summary>
    /// Gets or sets the min silence duration ms.
    /// </summary>
    public int MinSilenceDurationMs { get; set; }

    /// <summary>
    /// Gets or sets the conv filters.
    /// </summary>
    public int ConvFilters { get; set; }

    /// <summary>
    /// Gets or sets the lstm hidden dim.
    /// </summary>
    public int LstmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num lstm layers.
    /// </summary>
    public int NumLstmLayers { get; set; }

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
