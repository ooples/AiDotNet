using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for instance segmentation models that detect and mask individual object instances.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Instance segmentation finds each individual object in an image and provides
/// a pixel-level mask for each one. Unlike semantic segmentation (all cars are "car"), instance
/// segmentation distinguishes car #1 from car #2.
///
/// Models extending this base class: YOLOv9-Seg, YOLO11-Seg, YOLOv12-Seg, YOLO26-Seg, Mask2Former, MaskDINO.
/// </para>
/// </remarks>
public abstract class InstanceSegmentationBase<T> : SegmentationModelBase<T>, IInstanceSegmentation<T>
{
    private readonly int _maxInstances;
    private double _confidenceThreshold;
    private double _nmsThreshold;

    /// <inheritdoc/>
    public int MaxInstances => _maxInstances;

    /// <inheritdoc/>
    public double ConfidenceThreshold
    {
        get => _confidenceThreshold;
        set => _confidenceThreshold = value;
    }

    /// <inheritdoc/>
    public double NmsThreshold
    {
        get => _nmsThreshold;
        set => _nmsThreshold = value;
    }

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected InstanceSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses,
        int maxInstances = 100,
        double confidenceThreshold = 0.5,
        double nmsThreshold = 0.5)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _maxInstances = maxInstances;
        _confidenceThreshold = confidenceThreshold;
        _nmsThreshold = nmsThreshold;
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected InstanceSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses,
        int maxInstances = 100,
        double confidenceThreshold = 0.5,
        double nmsThreshold = 0.5)
        : base(architecture, onnxModelPath, numClasses)
    {
        _maxInstances = maxInstances;
        _confidenceThreshold = confidenceThreshold;
        _nmsThreshold = nmsThreshold;
    }

    /// <inheritdoc/>
    public abstract InstanceSegmentationResult<T> DetectInstances(Tensor<T> image);
}
