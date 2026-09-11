using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>Shared image, text-context, and output dimensions for vision-language models.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings describe the pictures and text supplied to a
/// model and the width of its embeddings. They do not describe or rebuild its internal layers.
/// For an ONNX model, the configured dimensions must match the loaded graphs.</para>
/// <para>Native tower settings belong to <see cref="VisionLanguageModelOptions"/>.
/// Inference-only models can use this input contract without advertising unused tower setters.</para>
/// </remarks>
public abstract class VisionLanguageInputOptions : ModelHyperparameterOptions
{
    /// <summary>Gets or sets the width of the shared image/text embedding space.</summary>
    /// <remarks><para><b>For Beginners:</b> An image and a sentence are each represented
    /// by this many numbers so that their embeddings can be compared.</para></remarks>
    public int EmbeddingDimension { get; set; }

    /// <summary>Gets or sets the maximum text sequence length, in tokens.</summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>Gets or sets the side length, in pixels, of the square input image.</summary>
    public int ImageSize { get; set; }

    /// <summary>Identifies which input dimensions the model consumes.</summary>
    [Flags]
    protected enum InputValidationRequirements
    {
        /// <summary>Only an embedding width is required.</summary>
        None = 0,
        /// <summary>A positive text context is required.</summary>
        Text = 1,
        /// <summary>A positive square-image size is required.</summary>
        Image = 2
    }

    /// <summary>Validates the shared embedding width and the selected input dimensions.</summary>
    /// <param name="requirements">The input geometry consumed by the model.</param>
    /// <exception cref="ArgumentException">A required dimension is zero or negative.</exception>
    protected void ValidateInputs(InputValidationRequirements requirements)
    {
        Require(EmbeddingDimension, nameof(EmbeddingDimension));
        if ((requirements & InputValidationRequirements.Text) != 0)
            Require(MaxSequenceLength, nameof(MaxSequenceLength));
        if ((requirements & InputValidationRequirements.Image) != 0)
            Require(ImageSize, nameof(ImageSize));
    }
}
