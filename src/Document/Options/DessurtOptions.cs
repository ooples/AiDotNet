using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the Dessurt document model.
/// </summary>
public class DessurtOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DessurtOptions"/> class carrying
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
    public DessurtOptions()
    {
        ImageSize = 1024;
        MaxSequenceLength = 512;
        EncoderDim = 1024;
        DecoderDim = 768;
        EncoderLayers = 24;
        DecoderLayers = 12;
        NumHeads = 16;
        VocabSize = 50265;
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
