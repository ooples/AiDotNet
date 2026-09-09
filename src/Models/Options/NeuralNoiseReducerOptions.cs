using AiDotNet.Audio.Enhancement;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for neural noise reduction models.
/// </summary>
public class NeuralNoiseReducerOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNoiseReducerOptions"/> class carrying
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
    public NeuralNoiseReducerOptions()
    {
        SampleRate = 16000;
        FftSize = 512;
        HopSize = 256;
        NumChannels = 1;
        EnhancementStrength = 0.8;
        NumStages = 4;
        BaseFilters = 32;
        BottleneckDim = 256;
    }


    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the fft size.
    /// </summary>
    public int FftSize { get; set; }

    /// <summary>
    /// Gets or sets the hop size.
    /// </summary>
    public int HopSize { get; set; }

    /// <summary>
    /// Gets or sets the num channels.
    /// </summary>
    public int NumChannels { get; set; }

    /// <summary>
    /// Gets or sets the enhancement strength.
    /// </summary>
    public double EnhancementStrength { get; set; }

    /// <summary>
    /// Gets or sets the num stages.
    /// </summary>
    public int NumStages { get; set; }

    /// <summary>
    /// Gets or sets the base filters.
    /// </summary>
    public int BaseFilters { get; set; }

    /// <summary>
    /// Gets or sets the bottleneck dim.
    /// </summary>
    public int BottleneckDim { get; set; }

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
