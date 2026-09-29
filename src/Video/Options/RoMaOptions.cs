using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RoMa (Robust Dense Matching) model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the RoMa (Robust Dense Matching) model for robust dense pixel correspondence.
/// Default values follow the original paper recommendations.</para>
/// </remarks>
public class RoMaOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RoMaOptions"/> class carrying
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
    public RoMaOptions()
    {
        NumFeatures = 64;
        NumLayers = 8;
    }


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
