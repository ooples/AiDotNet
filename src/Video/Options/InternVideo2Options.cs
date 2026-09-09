using AiDotNet.Models.Options;

using AiDotNet.Video.Understanding;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the InternVideo2 video understanding model.
/// </summary>
public class InternVideo2Options : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InternVideo2Options"/> class carrying
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
    public InternVideo2Options()
    {
        EmbedDim = 768;
        NumHeads = 12;
        NumEncoderLayers = 12;
        NumFrames = 8;
        PatchSize = 14;
    }


    /// <summary>
    /// Gets or sets the num encoder layers.
    /// </summary>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the patch size.
    /// </summary>
    public int PatchSize { get; set; }

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
