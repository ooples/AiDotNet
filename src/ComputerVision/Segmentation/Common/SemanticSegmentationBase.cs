using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for semantic segmentation models that assign a class label to every pixel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Semantic segmentation answers "what is this pixel?" for every pixel in an
/// image. This base class provides the shared infrastructure for models like SegFormer, SegNeXt,
/// InternImage, and DiffSeg — all of which produce a per-pixel class label map.
///
/// Extending this class gives you:
/// - Dual-mode support (native training + ONNX inference)
/// - Automatic class map extraction (argmax of logits)
/// - Probability map generation (softmax of logits)
/// - All common serialization and batch handling
/// </para>
/// </remarks>
public abstract class SemanticSegmentationBase<T> : SegmentationModelBase<T>, ISemanticSegmentation<T>
{
    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected SemanticSegmentationBase(
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
    protected SemanticSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses)
        : base(architecture, onnxModelPath, numClasses)
    {
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GetClassMap(Tensor<T> image)
    {
        var logits = Segment(image);
        return ArgmaxAlongClassDim(logits);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GetProbabilityMap(Tensor<T> image)
    {
        var logits = Segment(image);
        return SoftmaxAlongClassDim(logits);
    }

    /// <summary>
    /// Computes argmax along the class dimension (dim 0 for [C, H, W] or dim 1 for [B, C, H, W]).
    /// Delegates to <see cref="SegmentationTensorOps.ArgmaxAlongClassDim{T}"/> for the shared implementation.
    /// </summary>
    protected Tensor<T> ArgmaxAlongClassDim(Tensor<T> logits)
        => SegmentationTensorOps.ArgmaxAlongClassDim(logits);

    /// <summary>
    /// Computes softmax along the class dimension.
    /// Delegates to <see cref="SegmentationTensorOps.SoftmaxAlongClassDim{T}"/> for the shared implementation.
    /// </summary>
    protected Tensor<T> SoftmaxAlongClassDim(Tensor<T> logits)
        => SegmentationTensorOps.SoftmaxAlongClassDim(logits);
}
