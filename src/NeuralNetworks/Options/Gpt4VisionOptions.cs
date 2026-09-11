using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the Gpt4VisionNeuralNetwork.
/// </summary>
public class Gpt4VisionOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Gpt4VisionOptions"/> class carrying
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
    public Gpt4VisionOptions()
    {
        EmbeddingDimension = 4096;
        VisionDim = 1024;
        MaxSequenceLength = 2048;
        ContextWindowSize = 128000;
        ImageSize = 336;
        MaxImagesPerRequest = 10;
        HiddenDim = 4096;
        VisionLayers = 24;
        NumLmLayers = 32;
        NumHeads = 32;
        PatchSize = 14;
        VocabSize = 128256;
    }


    /// <summary>
    /// Gets or sets the context window size.
    /// </summary>
    public int ContextWindowSize { get; set; }

    /// <summary>
    /// Gets or sets the max images per request.
    /// </summary>
    public int MaxImagesPerRequest { get; set; }

    /// <summary>
    /// Gets or sets the number of native language-model blocks, using the same name
    /// as the Flamingo and LLaVA options. This does not resize a loaded ONNX graph.
    /// </summary>
    public int NumLmLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
        Require(VisionDim, nameof(VisionDim));
    }
}
