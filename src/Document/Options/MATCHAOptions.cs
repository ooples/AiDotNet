using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the MATCHA document model.
/// </summary>
public class MATCHAOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MATCHAOptions"/> class carrying
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
    public MATCHAOptions()
    {
        ImageSize = 2048;
        MaxSequenceLength = 512;
        EncoderDim = 1536;
        DecoderDim = 1536;
        EncoderLayers = 18;
        DecoderLayers = 18;
        NumHeads = 24;
        VocabSize = 50265;
        MaxPatchesPerImage = 4096;
    }


    /// <summary>
    /// Gets or sets the encoder dim.
    /// </summary>
    public int EncoderDim { get; set; }

    /// <summary>
    /// Gets or sets the decoder dim.
    /// </summary>
    public int DecoderDim { get; set; }

    /// <summary>
    /// Gets or sets the encoder layers.
    /// </summary>
    public int EncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the decoder layers.
    /// </summary>
    public int DecoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the max patches per image.
    /// </summary>
    public int MaxPatchesPerImage { get; set; }

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
