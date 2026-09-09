using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the Pix2Struct document model.
/// </summary>
public class Pix2StructOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Pix2StructOptions"/> class carrying
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
    public Pix2StructOptions()
    {
        ImageSize = 2048;
        PatchSize = 16;
        MaxPatches = 4096;
        MaxSequenceLength = 1024;
        HiddenDim = 1024;
        NumEncoderLayers = 18;
        NumDecoderLayers = 18;
        NumHeads = 16;
        VocabSize = 50000;
    }


    /// <summary>
    /// Gets or sets the max patches.
    /// </summary>
    public int MaxPatches { get; set; }

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
