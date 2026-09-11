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
    /// Gets or sets the number of native text-decoder transformer blocks.
    /// </summary>
    /// <value>A count of decoder blocks. Defaults to 12, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The decoder generates text using the image representation and
    /// preceding tokens. This controls its native depth; a loaded ONNX decoder keeps its own layers.</para>
    /// </remarks>
    public int NumDecoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the intermediate feature width of the native transformer's feed-forward sublayers.
    /// </summary>
    /// <value>The feed-forward width in features. Defaults to 3072, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Inside a transformer block, a small neural network expands
    /// each token to this width before projecting it back. A wider expansion uses more parameters
    /// and memory. This native construction setting does not resize loaded ONNX graphs.</para>
    /// </remarks>
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
        Require(VocabSize, nameof(VocabSize));
        Require(HiddenDim, nameof(HiddenDim));
        Require(NumEncoderLayers, nameof(NumEncoderLayers));
        Require(NumDecoderLayers, nameof(NumDecoderLayers));
        Require(NumHeads, nameof(NumHeads));
        Require(MlpDim, nameof(MlpDim));
    }
}
