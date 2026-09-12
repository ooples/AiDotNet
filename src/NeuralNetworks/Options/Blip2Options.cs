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

    /// <summary>Copies every BLIP-2 setting and its inherited configuration.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">The source is null.</exception>
    public Blip2Options(Blip2Options other) : base(other)
    {
        QformerHiddenDim = other.QformerHiddenDim;
        LmHiddenDim = other.LmHiddenDim;
        NumQformerLayers = other.NumQformerLayers;
        NumQueryTokens = other.NumQueryTokens;
        NumLmDecoderLayers = other.NumLmDecoderLayers;
        LanguageModelBackbone = other.LanguageModelBackbone;
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
    /// does not resize a loaded ONNX graph: an image-query export must produce exactly this
    /// many tokens. Text-only exports have no visual-query count and reject nondefault overrides.</para>
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
        Require(QformerHiddenDim, nameof(QformerHiddenDim));
        Require(VisionDim, nameof(VisionDim));
        Require(LmHiddenDim, nameof(LmHiddenDim));
        Require(NumQformerLayers, nameof(NumQformerLayers));
        Require(NumQueryTokens, nameof(NumQueryTokens));
        Require(NumHeads, nameof(NumHeads));
        Require(NumLmDecoderLayers, nameof(NumLmDecoderLayers));
        Require(VocabSize, nameof(VocabSize));
    }

    /// <summary>Validates host dimensions and rejects overrides of opaque native-only internals.</summary>
    internal void ValidateOnnx()
    {
        ValidateInputs(InputValidationRequirements.Text | InputValidationRequirements.Image);
        Require(VisionDim, nameof(VisionDim));
        Require(NumQueryTokens, nameof(NumQueryTokens));
        var defaults = new Blip2Options();
        RequireNativeDefaultForOnnx(QformerHiddenDim, defaults.QformerHiddenDim, nameof(QformerHiddenDim));
        RequireNativeDefaultForOnnx(LmHiddenDim, defaults.LmHiddenDim, nameof(LmHiddenDim));
        RequireNativeDefaultForOnnx(NumQformerLayers, defaults.NumQformerLayers, nameof(NumQformerLayers));
        RequireNativeDefaultForOnnx(NumHeads, defaults.NumHeads, nameof(NumHeads));
        RequireNativeDefaultForOnnx(NumLmDecoderLayers, defaults.NumLmDecoderLayers, nameof(NumLmDecoderLayers));
        RequireNativeDefaultForOnnx(PatchSize, defaults.PatchSize, nameof(PatchSize));
        RequireNativeDefaultForOnnx(VocabSize, defaults.VocabSize, nameof(VocabSize));
        RequireNativeDefaultForOnnx(Channels, defaults.Channels, nameof(Channels));
    }

    /// <summary>
    /// Gets or sets the language model backbone.
    /// </summary>
    /// <value>Defaults to <see cref="LanguageModelBackbone.OPT"/>, preserving the model's previous constructor default.</value>
    /// <remarks><para><b>For Beginners:</b> This selects the language-model family used
    /// for tokenizer and generation conventions. It does not convert or replace the supplied
    /// ONNX language graph; the graph and tokenizer must belong to a compatible family.</para></remarks>
    public LanguageModelBackbone LanguageModelBackbone { get; set; }
}
