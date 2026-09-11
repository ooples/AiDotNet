namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Input and embedding configuration for the ONNX-only ClipNeuralNetwork.
/// </summary>
/// <remarks>
/// <para>The three dimensions must match the supplied ONNX encoders; they do not resize
/// the graphs. The image path accepts three RGB channels. Patch sizes, vocabularies, tower
/// widths, heads, and layer counts are properties of those graphs, not configurable native
/// layers, so this options type does not expose setters for them.</para>
/// </remarks>
public class ClipOptions : VisionLanguageInputOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ClipOptions"/> class carrying
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
    public ClipOptions()
    {
        EmbeddingDimension = 512;
        MaxSequenceLength = 77;
        ImageSize = 224;
    }


    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateInputs(InputValidationRequirements.Text | InputValidationRequirements.Image);
    }
}
