using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the ImageBindNeuralNetwork.
/// </summary>
public class ImageBindOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ImageBindOptions"/> class carrying
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
    public ImageBindOptions()
    {
        EmbeddingDimension = 1024;
        MaxSequenceLength = 77;
        ImageSize = 224;
        AudioSampleRate = 16000;
        Channels = 3;
        PatchSize = 14;
        VocabSize = 49408;
        HiddenDim = 1280;
        NumEncoderLayers = 32;
        NumHeads = 16;
        AudioMaxDuration = 10;
        ImuTimesteps = 2000;
        NumVideoFrames = 2;
    }


    /// <summary>
    /// Gets or sets the audio sample rate.
    /// </summary>
    public int AudioSampleRate { get; set; }

    /// <summary>
    /// Gets or sets the audio max duration.
    /// </summary>
    public int AudioMaxDuration { get; set; }

    /// <summary>
    /// Gets or sets the imu timesteps.
    /// </summary>
    public int ImuTimesteps { get; set; }

    /// <summary>
    /// Gets or sets the num video frames.
    /// </summary>
    public int NumVideoFrames { get; set; }

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
