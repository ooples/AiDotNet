using AiDotNet.Models.Options;

using AiDotNet.Video.Generation;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the CogVideo generation model.
/// </summary>
public class CogVideoOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CogVideoOptions"/> class carrying
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
    public CogVideoOptions()
    {
        EmbedDim = 1024;
        NumLayers = 24;
        NumFrames = 16;
        NumTimesteps = 1000;
    }


    /// <summary>
    /// Gets or sets the num timesteps.
    /// </summary>
    public int NumTimesteps { get; set; }

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
