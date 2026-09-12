using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the TRIE document model.
/// </summary>
public class TRIEOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TRIEOptions"/> class carrying
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
    public TRIEOptions()
    {
        ImageSize = 512;
        VisualDim = 256;
        TextDim = 256;
        GraphDim = 256;
        NumEntityTypes = 10;
        MaxEntities = 100;
    }


    /// <summary>
    /// Gets or sets the visual dim.
    /// </summary>
    public int VisualDim { get; set; }

    /// <summary>
    /// Gets or sets the text dim.
    /// </summary>
    public int TextDim { get; set; }

    /// <summary>
    /// Gets or sets the graph dim.
    /// </summary>
    public int GraphDim { get; set; }

    /// <summary>
    /// Gets or sets the num entity types.
    /// </summary>
    public int NumEntityTypes { get; set; }

    /// <summary>
    /// Gets or sets the max entities.
    /// </summary>
    public int MaxEntities { get; set; }

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
