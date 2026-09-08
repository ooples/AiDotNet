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
    public int VisionHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the vision tower.
    /// </summary>
    public int NumVisionLayers { get; set; }

    /// <summary>
    /// Throws if a dimension every vision-language model requires has been left unset.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    protected void ValidateCore()
    {
        Require(EmbeddingDimension, nameof(EmbeddingDimension));
        Require(MaxSequenceLength, nameof(MaxSequenceLength));
        Require(ImageSize, nameof(ImageSize));
        Require(Channels, nameof(Channels));
    }
}
