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
    /// Gets or sets the nominal waveform rate used when sizing the native audio encoder.
    /// </summary>
    /// <value>A rate in samples per second. Defaults to 16000, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Together with <see cref="AudioMaxDuration"/>, this sets native
    /// audio sequence capacity. It does not resample waveforms or override the sample-rate
    /// argument supplied when encoding audio. A loaded ONNX graph retains its input contract.</para>
    /// </remarks>
    public int AudioSampleRate { get; set; }

    /// <summary>
    /// Gets or sets the audio duration used to size native audio sequence and position storage.
    /// </summary>
    /// <value>A duration in seconds. Defaults to 10, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The native encoder combines this duration with the configured
    /// audio rate when allocating capacity. It is not an automatic waveform-trimming operation
    /// and does not change the shape required by an ONNX audio encoder.</para>
    /// </remarks>
    public int AudioMaxDuration { get; set; }

    /// <summary>
    /// Gets or sets the number of inertial-sensor time steps used for native IMU sequence storage.
    /// </summary>
    /// <value>A count of sequential observations, not milliseconds. Defaults to 2000, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each observation contains the sensor readings for one time
    /// step. This count sizes the native IMU encoder and its positional embeddings; it does not
    /// specify a sensor sampling rate. IMU encoding is supported only in native mode.</para>
    /// </remarks>
    public int ImuTimesteps { get; set; }

    /// <summary>
    /// Gets or sets the number of frames selected for the native video embedding path.
    /// </summary>
    /// <value>A positive count of frames per clip. Defaults to 2, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Native video encoding selects uniformly from the supplied
    /// frames and repeats the last frame when needed. It does not decode a video file.
    /// The current ONNX video path uses the first supplied frame instead of this native frame count.</para>
    /// </remarks>
    public int NumVideoFrames { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
        Require(AudioSampleRate, nameof(AudioSampleRate));
        Require(AudioMaxDuration, nameof(AudioMaxDuration));
        Require(ImuTimesteps, nameof(ImuTimesteps));
        Require(NumVideoFrames, nameof(NumVideoFrames));
        Require(VocabSize, nameof(VocabSize));
        Require(HiddenDim, nameof(HiddenDim));
        Require(NumEncoderLayers, nameof(NumEncoderLayers));
        Require(NumHeads, nameof(NumHeads));
    }

    internal void ValidateOnnx()
    {
        ValidateInputs(InputValidationRequirements.Text | InputValidationRequirements.Image);
        var defaults = new ImageBindOptions();
        RequireNativeDefaultForOnnx(Channels, defaults.Channels, nameof(Channels));
        RequireNativeDefaultForOnnx(PatchSize, defaults.PatchSize, nameof(PatchSize));
        RequireNativeDefaultForOnnx(VocabSize, defaults.VocabSize, nameof(VocabSize));
        RequireNativeDefaultForOnnx(HiddenDim, defaults.HiddenDim, nameof(HiddenDim));
        RequireNativeDefaultForOnnx(NumEncoderLayers, defaults.NumEncoderLayers, nameof(NumEncoderLayers));
        RequireNativeDefaultForOnnx(NumHeads, defaults.NumHeads, nameof(NumHeads));
        RequireNativeDefaultForOnnx(AudioSampleRate, defaults.AudioSampleRate, nameof(AudioSampleRate));
        RequireNativeDefaultForOnnx(AudioMaxDuration, defaults.AudioMaxDuration, nameof(AudioMaxDuration));
        RequireNativeDefaultForOnnx(ImuTimesteps, defaults.ImuTimesteps, nameof(ImuTimesteps));
        RequireNativeDefaultForOnnx(NumVideoFrames, defaults.NumVideoFrames, nameof(NumVideoFrames));
    }
}
