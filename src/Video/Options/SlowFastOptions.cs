using AiDotNet.Models.Options;

using AiDotNet.Video.ActionRecognition;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the SlowFast video model.
/// </summary>
public class SlowFastOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SlowFastOptions"/> class carrying
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
    public SlowFastOptions()
    {
        NumClasses = 400;
        SlowFrames = 4;
        SlowChannels = 64;
        FastChannels = 8;
        Alpha = 8;
    }


    /// <summary>
    /// Gets or sets the slow frames.
    /// </summary>
    public int SlowFrames { get; set; }

    /// <summary>
    /// Gets or sets the slow channels.
    /// </summary>
    public int SlowChannels { get; set; }

    /// <summary>
    /// Gets or sets the fast channels.
    /// </summary>
    public int FastChannels { get; set; }

    /// <summary>
    /// Gets or sets the alpha.
    /// </summary>
    public int Alpha { get; set; }

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
