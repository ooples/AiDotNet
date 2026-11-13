using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the DiT document model.
/// </summary>
public class DiTOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DiTOptions"/> class carrying
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
    public DiTOptions()
    {
        NumClasses = 16;
        ImageSize = 224;
        PatchSize = 16;
        HiddenDim = 768;
        NumLayers = 12;
        NumHeads = 12;
        ModelSize = DiTModelSize.Base;
    }


    /// <summary>
    /// Gets or sets the published DiT variant this configuration describes (Li et al. 2022, Table 1).
    /// </summary>
    public DiTModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        // This model renders the page as an image, so it needs a size. The family base
        // cannot require this: 15 of the 29 document models work from text and layout
        // coordinates and have no image at all.
        Require(ImageSize, nameof(ImageSize));
        ValidateCore();
    }
}

/// <summary>The DiT variants released with the paper (Li et al. 2022, Sec. 3).</summary>
public enum DiTModelSize
{
    /// <summary>DiT-Base: ViT-B, 12 layers, hidden size 768, 12 heads.</summary>
    Base,

    /// <summary>DiT-Large: ViT-L, 24 layers, hidden size 1024, 16 heads.</summary>
    Large
}
