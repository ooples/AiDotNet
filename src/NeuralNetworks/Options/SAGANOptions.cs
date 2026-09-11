using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SAGAN neural network.
/// </summary>
public class SAGANOptions : GanOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SAGANOptions"/> class carrying
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
    public SAGANOptions()
    {
        NumClasses = 0;
        GeneratorChannels = 64;
        DiscriminatorChannels = 64;
        LatentSize = 128;
        ImageChannels = 3;
        ImageHeight = 64;
        ImageWidth = 64;
        InputType = InputType.TwoDimensional;
        InitialLearningRate = 0.0001;
    }


    /// <summary>
    /// Gets or sets the num classes.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the image height.
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Gets or sets the image width.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Gets or sets the input type.
    /// </summary>
    public InputType InputType { get; set; }

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
