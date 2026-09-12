using AiDotNet.Models.Options;

using AiDotNet.Video.Generation;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the StableVideoDiffusion model.
/// </summary>
public class StableVideoDiffusionOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="StableVideoDiffusionOptions"/> class carrying
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
    public StableVideoDiffusionOptions()
    {
        Variant = SVDModelVariant.SVD;
        NumFrames = 14;
        NumInferenceSteps = 25;
        GuidanceScale = 7.5;
        VaeChannels = 128;
        TextEncoderDim = 768;
        TextEncoderLayers = 12;
        TextEncoderHeads = 12;
    }


    /// <summary>
    /// Gets or sets the variant.
    /// </summary>
    public SVDModelVariant Variant { get; set; }

    /// <summary>
    /// Gets or sets the num inference steps.
    /// </summary>
    public int NumInferenceSteps { get; set; }

    /// <summary>
    /// Gets or sets the guidance scale.
    /// </summary>
    public double GuidanceScale { get; set; }

    /// <summary>
    /// Gets or sets the vae channels.
    /// </summary>
    public int VaeChannels { get; set; }

    /// <summary>
    /// Gets or sets the text encoder dim.
    /// </summary>
    public int TextEncoderDim { get; set; }

    /// <summary>
    /// Gets or sets the text encoder layers.
    /// </summary>
    public int TextEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the text encoder heads.
    /// </summary>
    public int TextEncoderHeads { get; set; }

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
