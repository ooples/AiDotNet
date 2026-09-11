using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Shared configuration for vision-language and multimodal models (CLIP, BLIP, BLIP-2,
/// Flamingo, LLaVA, ImageBind, GPT-4 Vision, VideoCLIP and relatives).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models read pictures and text together. The settings here
/// describe how big the pictures are, how they get chopped into patches, how much text the
/// model can read at once, and how wide its internal representations are. Each model's own
/// options class ships the values from its paper, so you do not normally set any of them.
/// </para>
/// <para>
/// Derived options classes assign their paper's values in their parameterless constructor.
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// </para>
/// </remarks>
public abstract class VisionLanguageModelOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the width of the shared embedding space that images and text are
    /// projected into.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both a picture and a sentence get turned into a list of
    /// numbers of this length, so they can be compared to each other.</para>
    /// </remarks>
    public int EmbeddingDimension { get; set; }

    /// <summary>
    /// Gets or sets the longest text sequence, in tokens, the model is configured to process.
    /// </summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the side length, in pixels, of the square image the model expects.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Input images are resized to this many pixels on each side —
    /// 224 and 336 are the common choices.</para>
    /// </remarks>
    public int ImageSize { get; set; }

    /// <summary>
    /// Gets or sets the side length, in pixels, of each square patch the image is divided into.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Vision transformers cut a picture into a grid of small
    /// squares and treat each one like a word. A 224-pixel image with 16-pixel patches becomes
    /// a 14 by 14 grid, so 196 "words".</para>
    /// </remarks>
    public int PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the number of colour channels in the input image.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 3 for ordinary colour images (red, green, blue).</para>
    /// </remarks>
    public int Channels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of distinct tokens the text tokenizer covers.
    /// </summary>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the width of the model's main hidden representation.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the text encoder.
    /// </summary>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the width of the vision tower's hidden representation, which is often
    /// wider than the shared embedding space.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The part of the model that looks at pictures usually works
    /// at a different, larger size than the shared space where pictures and text meet.</para>
    /// </remarks>
    public int VisionDim { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the vision tower.
    /// </summary>
    public int VisionLayers { get; set; }

    /// <summary>
    /// Validates the shared embedding, text context, and square-image dimensions.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    protected void ValidateCore()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.Image);
    }

    /// <summary>Identifies the input geometry a derived model actually consumes.</summary>
    [Flags]
    protected enum ValidationRequirements
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
    protected void ValidateCore(ValidationRequirements requirements)
    {
        Require(EmbeddingDimension, nameof(EmbeddingDimension));
        if ((requirements & ValidationRequirements.Text) != 0)
            Require(MaxSequenceLength, nameof(MaxSequenceLength));
        if ((requirements & (ValidationRequirements.Image | ValidationRequirements.PatchGeometry | ValidationRequirements.ExactPatchTiling)) != 0)
        {
            Require(ImageSize, nameof(ImageSize));
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
