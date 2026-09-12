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
    /// Gets or sets the audio sample rate.
    /// </summary>
    public int AudioSampleRate { get; set; }

    /// <summary>
    /// Gets or sets the video frame rate.
    /// </summary>
    public double VideoFrameRate { get; set; }

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
