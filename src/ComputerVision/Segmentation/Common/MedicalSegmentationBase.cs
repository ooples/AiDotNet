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
        if (queryImage is null) throw new ArgumentNullException(nameof(queryImage));
        if (supportImages is null) throw new ArgumentNullException(nameof(supportImages));
        if (supportMasks is null) throw new ArgumentNullException(nameof(supportMasks));

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
        if (volume is null) throw new ArgumentNullException(nameof(volume));
        if (volume.Rank < 4)
            throw new ArgumentException("Volume must have shape [C, D, H, W].", nameof(volume));
        if (overlap < 0 || overlap >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap must be in [0, 1).");
        if (patchSize.D <= 0 || patchSize.H <= 0 || patchSize.W <= 0)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch dimensions must be positive.");

        int channels = volume.Shape[0];
        int volD = volume.Shape[1], volH = volume.Shape[2], volW = volume.Shape[3];

        // If volume fits in a single patch, segment directly
        if (volD <= patchSize.D && volH <= patchSize.H && volW <= patchSize.W)
            return SegmentVolume(volume);

        int strideD = Math.Max(1, (int)(patchSize.D * (1.0 - overlap)));
        int strideH = Math.Max(1, (int)(patchSize.H * (1.0 - overlap)));
        int strideW = Math.Max(1, (int)(patchSize.W * (1.0 - overlap)));

        // Accumulation tensors for weighted averaging in overlap regions
        var accumOutput = new Tensor<T>([_numClasses, volD, volH, volW]);
        var accumCount = new double[volD * volH * volW];

        for (int d = 0; d < volD; d += strideD)
        {
            for (int h = 0; h < volH; h += strideH)
            {
                for (int w = 0; w < volW; w += strideW)
                {
                    int endD = Math.Min(d + patchSize.D, volD);
                    int endH = Math.Min(h + patchSize.H, volH);
                    int endW = Math.Min(w + patchSize.W, volW);
                    int pD = endD - d, pH = endH - h, pW = endW - w;

                    // Extract patch
                    var patch = new Tensor<T>([channels, pD, pH, pW]);
                    for (int c = 0; c < channels; c++)
                        for (int dd = 0; dd < pD; dd++)
                            for (int hh = 0; hh < pH; hh++)
                                for (int ww = 0; ww < pW; ww++)
                                    patch[c, dd, hh, ww] = volume[c, d + dd, h + hh, w + ww];

                    // Segment patch
                    var patchResult = SegmentVolume(patch);
                    if (patchResult.Probabilities == null) continue;

                    // Accumulate results
                    for (int cls = 0; cls < _numClasses && cls < patchResult.Probabilities.Shape[0]; cls++)
                        for (int dd = 0; dd < pD; dd++)
                            for (int hh = 0; hh < pH; hh++)
                                for (int ww = 0; ww < pW; ww++)
                                {
                                    accumOutput[cls, d + dd, h + hh, w + ww] = NumOps.Add(
                                        accumOutput[cls, d + dd, h + hh, w + ww],
                                        patchResult.Probabilities[cls, dd, hh, ww]);
                                    if (cls == 0)
                                        accumCount[(d + dd) * volH * volW + (h + hh) * volW + (w + ww)] += 1.0;
                                }
                }
            }
        }

        // Average overlapping regions
        for (int cls = 0; cls < _numClasses; cls++)
            for (int dd = 0; dd < volD; dd++)
                for (int hh = 0; hh < volH; hh++)
                    for (int ww = 0; ww < volW; ww++)
                    {
                        double count = accumCount[dd * volH * volW + hh * volW + ww];
                        if (count > 1.0)
                            accumOutput[cls, dd, hh, ww] = NumOps.FromDouble(
                                NumOps.ToDouble(accumOutput[cls, dd, hh, ww]) / count);
                    }

        return new MedicalSegmentationResult<T> { Probabilities = accumOutput };
    }
}
