using AiDotNet.Models.Options;

using AiDotNet.Video.Tracking;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the ByteTrack video tracking model.
/// </summary>
public class ByteTrackOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ByteTrackOptions"/> class carrying
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
    public ByteTrackOptions()
    {
        NumFeatures = 256;
        NumClasses = 1;
        HighThreshold = 0.6;
        LowThreshold = 0.1;
        MaxAge = 30;
    }


    /// <summary>
    /// Gets or sets the high threshold.
    /// </summary>
    public double HighThreshold { get; set; }

    /// <summary>
    /// Gets or sets the low threshold.
    /// </summary>
    public double LowThreshold { get; set; }

    /// <summary>
    /// Gets or sets the max age.
    /// </summary>
    public int MaxAge { get; set; }

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
