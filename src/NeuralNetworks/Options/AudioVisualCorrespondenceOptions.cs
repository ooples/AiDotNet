using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the AudioVisualCorrespondenceNetwork.
/// </summary>
public class AudioVisualCorrespondenceOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AudioVisualCorrespondenceOptions"/> class carrying
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
    public AudioVisualCorrespondenceOptions()
    {
        EmbeddingDimension = 512; // DEFAULT_EMBEDDING_DIM
        AudioSampleRate = 16000; // DEFAULT_SAMPLE_RATE
        VideoFrameRate = 25.0; // DEFAULT_FRAME_RATE
        NumEncoderLayers = 6;
    }


    /// <summary>
    /// Gets or sets the waveform sampling rate used to construct the model's log-mel front end.
    /// </summary>
    /// <value>A positive rate in samples per second. Defaults to 16000, preserving the implementation's former DEFAULT_SAMPLE_RATE constant.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells the audio front end how many waveform measurements
    /// represent one second, which determines the frequencies represented by its mel filters.
    /// It must match the input waveform's rate; assigning this option does not resample the waveform.</para>
    /// </remarks>
    public int AudioSampleRate { get; set; }

    /// <summary>
    /// Gets or sets the nominal rate associated with the supplied video frames.
    /// </summary>
    /// <value>A finite, positive rate in frames per second. Defaults to 25.0, preserving the implementation's former DEFAULT_FRAME_RATE constant.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This records the expected frame cadence as metadata. The
    /// current visual encoder averages the frames you supply; it does not resample or select
    /// them using this rate. Callers are responsible for preparing frames at the intended cadence.</para>
    /// </remarks>
    public double VideoFrameRate { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.None);
        Require(AudioSampleRate, nameof(AudioSampleRate));
        Require(VideoFrameRate, nameof(VideoFrameRate));
    }
}
