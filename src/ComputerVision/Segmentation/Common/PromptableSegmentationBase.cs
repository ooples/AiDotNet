using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for promptable segmentation models like SAM that accept user prompts
/// (points, boxes, masks) to segment specific objects.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Promptable segmentation lets you point at, draw a box around, or
/// describe what you want to segment. The model first encodes the image (which is expensive but
/// done once), then quickly processes each prompt against the encoded representation.
///
/// Models extending this base class: SAM, SAM 2, SAM-HQ, SegGPT, SEEM.
/// </para>
/// </remarks>
public abstract class PromptableSegmentationBase<T> : SegmentationModelBase<T>, IPromptableSegmentation<T>
{
    /// <summary>
    /// Cached image embedding from the most recent SetImage call.
    /// </summary>
    protected Tensor<T>? _imageEmbedding;

    /// <summary>
    /// Whether an image has been encoded and is ready for prompting.
    /// </summary>
    protected bool _imageSet;

    /// <inheritdoc/>
    public virtual bool SupportsPointPrompts => true;

    /// <inheritdoc/>
    public virtual bool SupportsBoxPrompts => true;

    /// <inheritdoc/>
    public virtual bool SupportsMaskPrompts => true;

    /// <inheritdoc/>
    public virtual bool SupportsTextPrompts => false;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected PromptableSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected PromptableSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses)
        : base(architecture, onnxModelPath, numClasses)
    {
    }

    /// <inheritdoc/>
    public virtual void SetImage(Tensor<T> image)
    {
        _imageEmbedding = EncodeImage(image);
        _imageSet = true;
    }

    /// <summary>
    /// Encodes an image into an embedding for subsequent prompted segmentation.
    /// Subclasses implement this with their specific image encoder (ViT, etc.).
    /// </summary>
    protected abstract Tensor<T> EncodeImage(Tensor<T> image);

    /// <inheritdoc/>
    public abstract PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels);

    /// <inheritdoc/>
    public abstract PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box);

    /// <inheritdoc/>
    public abstract PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask);

    /// <inheritdoc/>
    public abstract List<PromptedSegmentationResult<T>> SegmentEverything();

    /// <summary>
    /// Ensures an image has been set before prompting.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if no image has been set.</exception>
    protected void EnsureImageSet()
    {
        if (!_imageSet || _imageEmbedding is null)
        {
            throw new InvalidOperationException(
                "No image has been set. Call SetImage() before using prompt-based segmentation methods.");
        }
    }
}
