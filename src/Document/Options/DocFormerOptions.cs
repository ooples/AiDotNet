using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the DocFormer document model.
/// </summary>
public class DocFormerOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DocFormerOptions"/> class carrying
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
    public DocFormerOptions()
    {
        NumClasses = 16;
        ImageSize = 224;
        MaxSequenceLength = 512;
        HiddenDim = 768;
        NumLayers = 12;
        NumHeads = 12;
        VocabSize = 30522;
        SpatialDim = 96;
        MaxRelativePositions = 8;
    }


    /// <summary>
    /// Gets or sets the width of each spatial lookup table. DocFormer concatenates eight tables per axis
    /// (two edges, the extent, and five distances to the previous token), so this must be HiddenDim / 8.
    /// DocFormer-base: 96.
    /// </summary>
    public int SpatialDim { get; set; }

    /// <summary>
    /// Gets or sets the clip distance of the relative 1-D position tables in multi-modal attention. Reference: 8.
    /// </summary>
    public int MaxRelativePositions { get; set; }

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
        Require(MaxRelativePositions, nameof(MaxRelativePositions));
        if (SpatialDim * 8 != HiddenDim)
            throw new ArgumentException($"SpatialDim ({SpatialDim}) must be HiddenDim / 8 ({HiddenDim} / 8): DocFormer concatenates eight spatial tables per axis.", nameof(SpatialDim));
        ValidateCore();
    }
}
