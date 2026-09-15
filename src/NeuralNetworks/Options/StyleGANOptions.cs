using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the StyleGAN neural network.
/// </summary>
public class StyleGANOptions : GanOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="StyleGANOptions"/> class carrying
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
    public StyleGANOptions()
    {
        InitialLearningRate = 0.001;
        EnableStyleMixing = true;
        StyleMixingProbability = 0.9;
    }


    /// <summary>
    /// Gets or sets the enable style mixing.
    /// </summary>
    public bool EnableStyleMixing { get; set; }

    /// <summary>
    /// Gets or sets the style mixing probability.
    /// </summary>
    public double StyleMixingProbability { get; set; }

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
