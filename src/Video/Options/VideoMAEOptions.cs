using AiDotNet.Models.Options;

using AiDotNet.Video.ActionRecognition;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoMAE video model.
/// </summary>
public class VideoMAEOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VideoMAEOptions"/> class carrying
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
    public VideoMAEOptions()
    {
        NumClasses = 400;
        NumFrames = 16;
        NumFeatures = 768;
        MaskRatio = 0.9;
    }


    /// <summary>
    /// Gets or sets the mask ratio.
    /// </summary>
    public double MaskRatio { get; set; }

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
