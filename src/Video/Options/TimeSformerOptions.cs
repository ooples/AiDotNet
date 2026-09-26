using AiDotNet.Models.Options;

using AiDotNet.Video.ActionRecognition;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the TimeSformer video model.
/// </summary>
public class TimeSformerOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimeSformerOptions"/> class carrying
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
    public TimeSformerOptions()
    {
        NumClasses = 400;
        EmbedDim = 768;
        NumHeads = 12;
        NumLayers = 12;
        NumFrames = 8;
        PatchSize = 16;
        AttentionType = AttentionType.DividedSpaceTime;
    }


    /// <summary>
    /// Gets or sets the patch size.
    /// </summary>
    public int PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the attention type.
    /// </summary>
    public AttentionType AttentionType { get; set; }

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
