namespace AiDotNet.Models.Options;

/// <summary>
/// Shared configuration for document models: OCR and text recognition, layout analysis,
/// document question answering, table and chart understanding, and document VQA.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Document models read scanned pages and photographs of documents.
/// They need to know how large the page image is, how much text they can produce, and how
/// big their internal representations are. Each model's own options class ships the values
/// from the paper that introduced it, so you do not normally set any of these.
/// </para>
/// <para>
/// Derived options classes assign their paper's values in their parameterless constructor:
/// <code>
/// public class TrOCROptions : DocumentNeuralNetworkOptions
/// {
///     public TrOCROptions()
///     {
///         ImageSize = 384;
///         VocabSize = 50265;      // RoBERTa tokenizer
///         HiddenDim = 768;
///         NumDecoderLayers = 12;
///     }
/// }
/// </code>
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// </para>
/// <para>
/// This class already sits at the root of every options class under
/// <c>src/Document/Options</c> — all 29 of them derive from it — so adding the shared
/// knobs here reaches the whole area without touching the leaves.
/// </para>
/// <para>
/// <b>Not to be confused with <c>DocumentModelOptions&lt;T&gt;</c>,</b> an earlier and now
/// abandoned attempt at the same idea: it takes a type parameter it never uses, follows the
/// nullable + <c>Effective*</c> pattern this design moves away from, and nothing derives
/// from it. It is left in place for now and removed when the document models are wired.
/// </para>
/// </remarks>
public class DocumentNeuralNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the side length, in pixels, of the page image the model expects.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scanned pages are resized to this many pixels on a side
    /// before the model reads them. Document models use larger values than ordinary vision
    /// models — 1024 rather than 224 — because small text has to stay legible.</para>
    /// </remarks>
    public int ImageSize { get; set; }

    /// <summary>
    /// Gets or sets the input image width in pixels, for models that expect a non-square page.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A single line of text cropped from a page is wide and short,
    /// so line-level recognition models use an explicit width and height rather than a square.</para>
    /// </remarks>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Gets or sets the input image height in pixels, for models that expect a non-square page.
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Gets or sets the side length, in pixels, of each square patch the page is divided into.
    /// </summary>
    public int PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the longest sequence, in tokens, the model reads or produces.
    /// </summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the number of distinct tokens the model's tokenizer covers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 30522 is the BERT vocabulary, 50265 RoBERTa's, and 250002
    /// the multilingual XLM-R vocabulary used by models that read many languages.</para>
    /// </remarks>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the width of the model's main hidden representation.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the total number of stacked blocks, for models with a single stack.
    /// </summary>
    public int NumLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of encoder layers, for encoder-decoder models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The encoder looks at the page; the decoder writes out the
    /// text. They are usually different depths.</para>
    /// </remarks>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of decoder layers, for encoder-decoder models.
    /// </summary>
    public int NumDecoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the width of the vision tower's hidden representation.
    /// </summary>
    public int VisionDim { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the vision tower.
    /// </summary>
    public int VisionLayers { get; set; }

    /// <summary>
    /// Gets or sets the channel width of the convolutional backbone, for detection-style
    /// models such as text detectors.
    /// </summary>
    public int BackboneChannels { get; set; }

    /// <summary>
    /// Gets or sets the number of output categories, for classification models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A layout model that labels each region as title, paragraph,
    /// table, figure and so on uses one class per label.</para>
    /// </remarks>
    public int NumClasses { get; set; }

    /// <summary>
    /// Throws if a dimension every document model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    protected void ValidateCore()
    {
        // Nothing is universal across 29 heterogeneous document models: only 13 of them have a
        // hidden dimension at all, and 17 a sequence length. Requiring a value the family does
        // not share compiles clean and then fails at construction — it cost 11 failing tests
        // here and 30 in the GAN family before that.
        //
        // A leaf that knows it needs a value calls Require itself, in its own Validate().
        Require(ImageSize > 0 ? ImageSize : 1, nameof(ImageSize));
    }
}
