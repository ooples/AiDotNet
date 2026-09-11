using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the BlipNeuralNetwork.
/// </summary>
public class BlipOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="BlipOptions"/> class carrying
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
    public BlipOptions()
    {
        EmbeddingDimension = 256;
        MaxSequenceLength = 35;
        ImageSize = 384;
        Channels = 3;
        PatchSize = 16;
        VocabSize = 30522;
        HiddenDim = 768;
        NumEncoderLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        MlpDim = 3072;
    }


    /// <summary>
    /// Gets or sets the num decoder layers.
    /// </summary>
    public int NumDecoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the mlp dim.
    /// </summary>
    public int MlpDim { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
    }
}
