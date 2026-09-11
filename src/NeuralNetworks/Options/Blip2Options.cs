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
    /// Gets or sets the feature width of the native query transformer that extracts image information.
    /// </summary>
    /// <value>The query-transformer width in features. Defaults to 768, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each learned image query carries this many numbers while it
    /// interacts with image features. This width is separate from the vision and language-model
    /// widths. It does not resize the query transformer loaded from an ONNX graph.</para>
    /// </remarks>
    public int QformerHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the native language-model feature width used after query-to-language projection.
    /// </summary>
    /// <value>The language-model width in features. Defaults to 2560, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Image-query information is projected into this width before
    /// the native language decoder uses it. It is not the query-transformer width, and changing
    /// it does not rebuild or resize an ONNX language model.</para>
    /// </remarks>
    public int LmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of native query-transformer blocks.
    /// </summary>
    /// <value>A count of query-transformer blocks. Defaults to 12, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These blocks repeatedly refine the learned queries using
    /// image information. The count controls native construction; a loaded ONNX query transformer
    /// retains the depth stored in its graph.</para>
    /// </remarks>
    public int NumQformerLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of learned visual-query tokens allocated by the native query transformer.
    /// </summary>
    /// <value>A positive count of query tokens. Defaults to 32, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are trainable queries that summarize image information,
    /// not words in a caption. More queries increase the native query-state size. This option
    /// does not change the query-token layout expected by a loaded ONNX graph.</para>
    /// </remarks>
    public int NumQueryTokens { get; set; }

    /// <summary>
    /// Gets or sets the number of native language-model decoder blocks.
    /// </summary>
    /// <value>A count of language decoder blocks. Defaults to 6, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These blocks turn projected image-query features and earlier
    /// text into the next-token representation. They control the library's native decoder,
    /// not the depth of an externally loaded ONNX language model.</para>
    /// </remarks>
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
