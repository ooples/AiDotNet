using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the LLaVANeuralNetwork.
/// </summary>
public class LLaVAOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LLaVAOptions"/> class carrying
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
    public LLaVAOptions()
    {
        EmbeddingDimension = 4096;
        MaxSequenceLength = 2048;
        ImageSize = 336;
        Channels = 3;
        PatchSize = 14;
        VocabSize = 32000;
        VisionDim = 1024;
        VisionLayers = 24;
        NumLmLayers = 32;
        NumHeads = 16;
        LanguageModelBackbone = LanguageModelBackbone.LLaMA;
        VisionEncoderType = "clip-vit-l";
    }


    /// <summary>
    /// Gets or sets the num lm layers.
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
    }

    /// <summary>
    /// Gets or sets the language model backbone.
    /// </summary>
    public LanguageModelBackbone LanguageModelBackbone { get; set; }

    /// <summary>
    /// Gets or sets the vision encoder type.
    /// </summary>
    public string VisionEncoderType { get; set; }
}
