using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the CycleGAN neural network.
/// </summary>
public class CycleGANOptions : GanOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CycleGANOptions"/> class carrying
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
    public CycleGANOptions()
    {
        CycleConsistencyLambda = 10.0;
        IdentityLambda = 5.0;
    }


    /// <summary>
    /// Gets or sets the cycle consistency lambda.
    /// </summary>
    public double CycleConsistencyLambda { get; set; }

    /// <summary>
    /// Gets or sets the identity lambda.
    /// </summary>
    public double IdentityLambda { get; set; }

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
