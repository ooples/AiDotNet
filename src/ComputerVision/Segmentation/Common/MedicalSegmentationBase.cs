using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for medical image segmentation models handling 2D slices and 3D volumes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Medical segmentation helps doctors by automatically outlining organs,
/// tumors, and other structures in medical images. These models handle special requirements
/// like 3D volumetric processing (CT/MRI scans are stacks of slices), multi-modal imaging,
/// and the very high accuracy needed for clinical use.
///
/// Models extending this base class: nnU-Net, TransUNet, Swin-UNETR, MedSAM, MedSAM 2, MedNeXt.
/// </para>
/// </remarks>
public abstract class MedicalSegmentationBase<T> : SegmentationModelBase<T>, IMedicalSegmentation<T>
{
    private readonly List<string> _supportedModalities;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedModalities => _supportedModalities;

    /// <inheritdoc/>
    public virtual bool Supports3D => true;

    /// <inheritdoc/>
    public virtual bool Supports2D => true;

    /// <inheritdoc/>
    public virtual bool SupportsFewShot => false;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected MedicalSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses,
        IEnumerable<string>? supportedModalities = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _supportedModalities = supportedModalities?.ToList() ?? ["CT", "MRI"];
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected MedicalSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses,
        IEnumerable<string>? supportedModalities = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _supportedModalities = supportedModalities?.ToList() ?? ["CT", "MRI"];
    }

    /// <inheritdoc/>
    public abstract MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice);

    /// <inheritdoc/>
    public abstract MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume);

    /// <inheritdoc/>
    public virtual MedicalSegmentationResult<T> SegmentFewShot(
        Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
    {
        if (!SupportsFewShot)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support few-shot segmentation. " +
                "Use a model like UniverSeg or MedSAM that supports the SupportsFewShot property.");
        }

        return SegmentFewShotInternal(queryImage, supportImages, supportMasks);
    }

    /// <summary>
    /// Model-specific few-shot segmentation. Only called when SupportsFewShot is true.
    /// </summary>
    protected virtual MedicalSegmentationResult<T> SegmentFewShotInternal(
        Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
    {
        throw new NotSupportedException("Few-shot segmentation not implemented by this model.");
    }

    /// <summary>
    /// Applies sliding window inference for 3D volumes that exceed GPU memory.
    /// </summary>
    /// <param name="volume">Full 3D volume [C, D, H, W].</param>
    /// <param name="patchSize">Size of each sliding window patch (depth, height, width).</param>
    /// <param name="overlap">Overlap fraction between adjacent patches (0.0 to 1.0).</param>
    /// <returns>Aggregated segmentation result for the entire volume.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Medical volumes (like full CT scans) are often too large to process
    /// at once. Sliding window inference breaks the volume into overlapping patches, segments
    /// each patch, then stitches the results back together. Overlapping regions are averaged
    /// for smoother boundaries.
    /// </para>
    /// </remarks>
    protected virtual MedicalSegmentationResult<T> SlidingWindowInference(
        Tensor<T> volume,
        (int D, int H, int W) patchSize,
        double overlap = 0.5)
    {
        // Default implementation: segment entire volume at once
        // Subclasses can override for memory-efficient patch-based inference
        return SegmentVolume(volume);
    }
}
