using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for video segmentation models that track and segment objects across frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Video segmentation extends image segmentation to temporal sequences.
/// The key challenge is maintaining consistent object identity across frames — tracking objects
/// as they move, get occluded, or change appearance.
///
/// Models extending this base class: SAM 2, Cutie, XMem, DEVA, EfficientTAM, UniVS.
/// </para>
/// </remarks>
public abstract class VideoSegmentationBase<T> : SegmentationModelBase<T>, IVideoSegmentation<T>
{
    private readonly int _maxTrackedObjects;

    /// <summary>
    /// Current frame index in the video sequence.
    /// </summary>
    protected int _currentFrameIndex;

    /// <summary>
    /// Whether tracking has been initialized with first-frame masks.
    /// </summary>
    protected bool _trackingInitialized;

    /// <summary>
    /// Object IDs currently being tracked.
    /// </summary>
    protected List<int> _trackedObjectIds = [];

    /// <inheritdoc/>
    public int MaxTrackedObjects => _maxTrackedObjects;

    /// <inheritdoc/>
    public virtual bool SupportsStreaming => true;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected VideoSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses,
        int maxTrackedObjects = 32)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _maxTrackedObjects = maxTrackedObjects;
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected VideoSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses,
        int maxTrackedObjects = 32)
        : base(architecture, onnxModelPath, numClasses)
    {
        _maxTrackedObjects = maxTrackedObjects;
    }

    /// <inheritdoc/>
    public virtual void InitializeTracking(Tensor<T> frame, Tensor<T> masks, int[]? objectIds = null)
    {
        _currentFrameIndex = 0;
        _trackingInitialized = true;

        if (objectIds != null)
        {
            _trackedObjectIds = [.. objectIds];
        }
        else
        {
            // Assign sequential IDs
            int numObjects = masks.Shape[0];
            _trackedObjectIds = Enumerable.Range(1, numObjects).ToList();
        }

        InitializeTrackingInternal(frame, masks, [.. _trackedObjectIds]);
    }

    /// <summary>
    /// Model-specific tracking initialization with first-frame features and masks.
    /// </summary>
    protected abstract void InitializeTrackingInternal(Tensor<T> frame, Tensor<T> masks, int[] objectIds);

    /// <inheritdoc/>
    public virtual VideoSegmentationResult<T> PropagateToFrame(Tensor<T> frame)
    {
        if (!_trackingInitialized)
        {
            throw new InvalidOperationException(
                "Tracking has not been initialized. Call InitializeTracking() with a first-frame mask before propagating.");
        }

        _currentFrameIndex++;
        return PropagateToFrameInternal(frame, _currentFrameIndex);
    }

    /// <summary>
    /// Model-specific mask propagation to the next frame.
    /// </summary>
    protected abstract VideoSegmentationResult<T> PropagateToFrameInternal(Tensor<T> frame, int frameIndex);

    /// <inheritdoc/>
    public abstract void AddCorrection(int objectId, Tensor<T> correctionMask);

    /// <inheritdoc/>
    public virtual void ResetTracking()
    {
        _currentFrameIndex = 0;
        _trackingInitialized = false;
        _trackedObjectIds.Clear();
        ResetTrackingInternal();
    }

    /// <summary>
    /// Model-specific cleanup of tracking memory and state.
    /// </summary>
    protected abstract void ResetTrackingInternal();
}
