using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Shared native-tower configuration for vision-language and multimodal models (BLIP, BLIP-2,
/// Flamingo, LLaVA, ImageBind, GPT-4 Vision, VideoCLIP and relatives).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models read pictures and text together. The settings here
/// describe how big the pictures are, how they get chopped into patches, how much text the
/// model can read at once, and how wide its internal representations are. Each model's own
/// options class supplies its shipped defaults.
/// </para>
/// <para>
/// Derived options classes assign their shipped values in their parameterless constructor.
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// ONNX-only input contracts use <see cref="VisionLanguageInputOptions"/> instead. Native
/// tower settings do not resize a loaded ONNX graph.
/// </para>
/// </remarks>
public abstract class VisionLanguageModelOptions : VisionLanguageInputOptions
{
    /// <summary>Initializes the shared native settings; concrete options supply model defaults.</summary>
    protected VisionLanguageModelOptions() { }

    /// <summary>Copies every native setting and the inherited input/model configuration.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">The source is null.</exception>
    protected VisionLanguageModelOptions(VisionLanguageModelOptions other) : base(other)
    {
        PatchSize = other.PatchSize;
        Channels = other.Channels;
        VocabSize = other.VocabSize;
        NumHeads = other.NumHeads;
        HiddenDim = other.HiddenDim;
        NumEncoderLayers = other.NumEncoderLayers;
        VisionDim = other.VisionDim;
        VisionLayers = other.VisionLayers;
    }

    /// <summary>Rejects a native-only override that cannot reconfigure a loaded ONNX graph.</summary>
    /// <typeparam name="TValue">The option's value type.</typeparam>
    /// <param name="value">The requested value.</param>
    /// <param name="defaultValue">The unchanged native default.</param>
    /// <param name="propertyName">The native-only property name.</param>
    /// <exception cref="ArgumentException">The caller requested a native-only override.</exception>
    internal void RequireNativeDefaultForOnnx<TValue>(TValue value, TValue defaultValue, string propertyName)
    {
        if (!EqualityComparer<TValue>.Default.Equals(value, defaultValue))
            throw new ArgumentException($"{GetType().Name}.{propertyName} configures native layers, not a loaded ONNX graph. " +
                "Use the native constructor to change this setting; ONNX architecture is defined by the supplied graphs.", OptionsParameterName);
    }

    /// <summary>
    /// Gets or sets the side length, in pixels, of each square patch the image is divided into.
    /// </summary>
    /// <value>A pixel count supplied by the concrete model's options constructor, preserving its previous patch default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Vision transformers cut a picture into a grid of small
    /// squares and treat each one like a word. A 224-pixel image with 16-pixel patches becomes
    /// a 14 by 14 grid, so 196 "words".</para>
    /// </remarks>
    public int PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the number of colour channels in the input image.
    /// </summary>
    /// <value>Defaults to 3, preserving the shared RGB input convention; concrete options may specialize it.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> 3 for ordinary colour images (red, green, blue).</para>
    /// </remarks>
    public int Channels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of distinct tokens the text tokenizer covers.
    /// </summary>
    /// <value>A vocabulary-entry count supplied by the concrete model's previous constructor default.</value>
    /// <remarks><para><b>For Beginners:</b> Native token embeddings and output heads allocate
    /// one vocabulary entry per supported token. This must agree with the native tokenizer;
    /// it does not resize a vocabulary inside a loaded ONNX graph.</para></remarks>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>A head count assigned by the concrete model's options constructor from its previous default.</value>
    /// <remarks><para><b>For Beginners:</b> Attention splits features among these parallel
    /// groups. The native model's attention widths must support the selected head count.</para></remarks>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the width of the model's main hidden representation.
    /// </summary>
    /// <value>A feature count assigned by concrete options when the model consumes this setting; otherwise it remains zero.</value>
    /// <remarks><para><b>For Beginners:</b> This controls the internal representation, not
    /// necessarily the final shared embedding. Larger widths allocate more native weights.</para></remarks>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the text encoder.
    /// </summary>
    /// <value>A block count assigned by concrete options from the model's previous constructor default when consumed.</value>
    /// <remarks><para><b>For Beginners:</b> More encoder blocks add sequential processing
    /// and parameters. Whether zero disables an optional stack is specified by the concrete model.</para></remarks>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the width of the vision tower's hidden representation, which is often
    /// wider than the shared embedding space.
    /// </summary>
    /// <value>A feature count assigned by each concrete options constructor from its previous vision-tower default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The part of the model that looks at pictures usually works
    /// at a different, larger size than the shared space where pictures and text meet.</para>
    /// </remarks>
    public int VisionDim { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the vision tower.
    /// </summary>
    /// <value>A block count assigned by the concrete options constructor from the model's previous default when consumed.</value>
    /// <remarks><para><b>For Beginners:</b> These native blocks refine image features before
    /// they are combined with text. The count cannot change a tower already stored in an ONNX graph.</para></remarks>
    public int VisionLayers { get; set; }

    /// <summary>
    /// Validates the shared embedding, text context, and square-image dimensions.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    internal void ValidateCore()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.Image);
    }

    /// <summary>Identifies the input geometry a derived model actually consumes.</summary>
    [Flags]
    internal enum ValidationRequirements
    {
        /// <summary>Only the shared embedding is required.</summary>
        None = 0,
        /// <summary>A bounded text-token context is required.</summary>
        Text = 1,
        /// <summary>A square image and channel count are required.</summary>
        Image = 2,
        /// <summary>Positive patches must produce at least one complete image patch; remainder cropping is supported.</summary>
        PatchGeometry = 4,
        /// <summary>The model additionally requires patches to tile the image exactly.</summary>
        ExactPatchTiling = 8
    }

    /// <summary>Validates only the dimensions consumed by the selected model paths.</summary>
    /// <param name="requirements">The model's input-geometry requirements.</param>
    /// <exception cref="ArgumentException">A required dimension or patch tiling is invalid.</exception>
    /// <remarks>Audio-only or feature-input paths do not acquire image or token requirements
    /// merely because their options share this base. Patch validation precedes every division.</remarks>
    internal void ValidateCore(ValidationRequirements requirements)
    {
        var inputs = InputValidationRequirements.None;
        if ((requirements & ValidationRequirements.Text) != 0)
            inputs |= InputValidationRequirements.Text;
        if ((requirements & (ValidationRequirements.Image | ValidationRequirements.PatchGeometry | ValidationRequirements.ExactPatchTiling)) != 0)
            inputs |= InputValidationRequirements.Image;
        ValidateInputs(inputs);
        if ((inputs & InputValidationRequirements.Image) != 0)
        {
            Require(Channels, nameof(Channels));
        }
        if ((requirements & (ValidationRequirements.PatchGeometry | ValidationRequirements.ExactPatchTiling)) != 0)
        {
            Require(PatchSize, nameof(PatchSize));
            if (ImageSize < PatchSize)
            {
                throw new ArgumentException(
                    $"{GetType().Name}.{nameof(ImageSize)} ({ImageSize}) must contain at least one complete " +
                    $"{GetType().Name}.{nameof(PatchSize)} ({PatchSize}) patch.",
                    OptionsParameterName);
            }
            if ((requirements & ValidationRequirements.ExactPatchTiling) != 0 && ImageSize % PatchSize != 0)
            {
                throw new ArgumentException(
                    $"{GetType().Name}.{nameof(ImageSize)} ({ImageSize}) must be evenly divisible by " +
                    $"{GetType().Name}.{nameof(PatchSize)} ({PatchSize}).",
                    OptionsParameterName);
            }
        }
    }
}
