using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the Blip2NeuralNetwork.
/// </summary>
public class Blip2Options : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Blip2Options"/> class carrying
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
    public Blip2Options()
    {
        EmbeddingDimension = 256;
        MaxSequenceLength = 32;
        ImageSize = 224;
        Channels = 3;
        PatchSize = 14;
        VocabSize = 30522;
        QformerHiddenDim = 768;
        VisionDim = 1408;
        LmHiddenDim = 2560;
        NumQformerLayers = 12;
        NumQueryTokens = 32;
        NumHeads = 12;
        NumLmDecoderLayers = 6;
        LanguageModelBackbone = LanguageModelBackbone.OPT;
    }


    /// <summary>
    /// Gets or sets the qformer hidden dim.
    /// </summary>
    public int QformerHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the lm hidden dim.
    /// </summary>
    public int LmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num qformer layers.
    /// </summary>
    public int NumQformerLayers { get; set; }

    /// <summary>
    /// Gets or sets the num query tokens.
    /// </summary>
    public int NumQueryTokens { get; set; }

    /// <summary>
    /// Gets or sets the num lm decoder layers.
    /// </summary>
    public int NumLmDecoderLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.ExactPatchTiling);
    }

    /// <summary>
    /// Gets or sets the language model backbone.
    /// </summary>
    public LanguageModelBackbone LanguageModelBackbone { get; set; }
}
