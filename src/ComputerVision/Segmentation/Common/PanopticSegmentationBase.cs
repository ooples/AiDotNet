using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for panoptic segmentation models that unify semantic and instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Panoptic segmentation gives you the most complete picture of a scene by
/// combining semantic segmentation (labeling regions like "road", "sky") with instance segmentation
/// (distinguishing individual objects like "car #1", "car #2").
///
/// Models extending this base class: Mask2Former, kMaX-DeepLab, OneFormer, ODISE.
/// </para>
/// </remarks>
public abstract class PanopticSegmentationBase<T> : SegmentationModelBase<T>, IPanopticSegmentation<T>
{
    private readonly int _numStuffClasses;
    private readonly int _numThingClasses;

    /// <inheritdoc/>
    public int NumStuffClasses => _numStuffClasses;

    /// <inheritdoc/>
    public int NumThingClasses => _numThingClasses;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected PanopticSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses,
        int numStuffClasses,
        int numThingClasses)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        if (numStuffClasses + numThingClasses != numClasses)
            throw new ArgumentException(
                $"numStuffClasses ({numStuffClasses}) + numThingClasses ({numThingClasses}) must equal numClasses ({numClasses}).");
        _numStuffClasses = numStuffClasses;
        _numThingClasses = numThingClasses;
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected PanopticSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses,
        int numStuffClasses,
        int numThingClasses)
        : base(architecture, onnxModelPath, numClasses)
    {
        if (numStuffClasses + numThingClasses != numClasses)
            throw new ArgumentException(
                $"numStuffClasses ({numStuffClasses}) + numThingClasses ({numThingClasses}) must equal numClasses ({numClasses}).");
        _numStuffClasses = numStuffClasses;
        _numThingClasses = numThingClasses;
    }

    /// <inheritdoc/>
    public abstract PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image);
}
