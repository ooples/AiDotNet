using AiDotNet.Models.Options;

using AiDotNet.Video.Inpainting;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the ProPainter video inpainting model.
/// </summary>
public class ProPainterOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ProPainterOptions"/> class carrying
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
    public ProPainterOptions()
    {
        NumFeatures = 128;
        NumTransformerBlocks = 6;
        NumHeads = 8;
    }


    /// <summary>
    /// Gets or sets the num transformer blocks.
    /// </summary>
    public int NumTransformerBlocks { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();

        // The transformer splits its NumFeatures * 4 channels across NumHeads heads; a width that does not divide
        // evenly would leave the remainder channels unattended (zero-filled), so it is rejected.
        int width = NumFeatures * 4;
        if (NumFeatures > 0 && NumHeads > 0 && width % NumHeads != 0)
        {
            throw new ArgumentException(
                $"NumFeatures * 4 ({width}) must be a multiple of NumHeads ({NumHeads}).", nameof(NumHeads));
        }
    }
}
