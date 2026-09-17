using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the ProgressiveGAN.
/// </summary>
public class ProgressiveGANOptions : GanOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ProgressiveGANOptions"/> class carrying
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
    public ProgressiveGANOptions()
    {
        MaxResolutionLevel = 6;
        GeneratorChannels = 512;
        LatentSize = 512;
        ImageChannels = 3;
        InputType = InputType.TwoDimensional;
        InitialLearningRate = 0.001; // DefaultLearningRate
        LearningRateDecay = 0.9999; // DefaultLearningRateDecay
    }


    /// <summary>
    /// Gets or sets the max resolution level.
    /// </summary>
    public int MaxResolutionLevel { get; set; }

    /// <summary>
    /// Gets or sets the input type.
    /// </summary>
    public InputType InputType { get; set; }

    /// <summary>
    /// Gets or sets the learning rate decay.
    /// </summary>
    public double LearningRateDecay { get; set; }

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
