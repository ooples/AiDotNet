using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for optical flow estimation models that compute dense pixel-wise motion between frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Optical flow estimation computes per-pixel motion vectors between two consecutive frames.
/// This base class provides:
///
/// - Flow field output (dense 2D displacement vectors)
/// - Multi-scale iterative refinement support
/// - Forward-backward consistency checking
/// - Flow visualization utilities
///
/// Derived classes implement specific architectures like RAFT, FlowFormer, SEA-RAFT, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Optical flow tells you how each pixel moved between two frames.
/// It's like tracking every single point in the image. The output is a "flow field" where
/// each position stores (dx, dy) - how far that pixel moved horizontally and vertically.
/// This is useful for video stabilization, frame interpolation, action recognition, and more.
/// </para>
/// </remarks>
public abstract class OpticalFlowBase<T> : VideoNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the number of iterative refinement steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Many modern flow models (RAFT, FlowFormer) use iterative refinement where
    /// the flow estimate is progressively improved. More iterations generally
    /// improve quality but increase computation.
    /// Common values: 6, 12, 24, 32.
    /// </para>
    /// </remarks>
    public int NumIterations { get; protected set; } = 12;

    /// <summary>
    /// Gets whether this model supports multi-scale processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multi-scale models build an image pyramid and estimate flow at multiple resolutions,
    /// propagating coarse estimates to finer levels. This helps capture large motions.
    /// </para>
    /// </remarks>
    public bool SupportsMultiScale { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the OpticalFlowBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected OpticalFlowBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Estimates optical flow between two frames.
    /// </summary>
    /// <param name="frame0">First (reference) frame [channels, height, width].</param>
    /// <param name="frame1">Second (target) frame [channels, height, width].</param>
    /// <returns>Flow field [2, height, width] where channel 0 is horizontal (dx) and channel 1 is vertical (dy).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how each pixel in frame0 moved to reach frame1.
    /// The output has 2 channels: dx (horizontal movement) and dy (vertical movement).
    /// Positive dx means the pixel moved right, positive dy means it moved down.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> EstimateFlow(Tensor<T> frame0, Tensor<T> frame1);

    /// <summary>
    /// Estimates optical flow at multiple scales for handling large motions.
    /// Override this in derived classes to provide actual multi-scale pyramid estimation.
    /// The default implementation returns only the full-scale flow as a single-element list.
    /// </summary>
    /// <param name="frame0">First frame [channels, height, width].</param>
    /// <param name="frame1">Second frame [channels, height, width].</param>
    /// <param name="numLevels">Number of pyramid levels.</param>
    /// <returns>List of flow fields from coarsest to finest resolution.</returns>
    public virtual List<Tensor<T>> EstimateFlowMultiScale(Tensor<T> frame0, Tensor<T> frame1, int numLevels = 4)
    {
        // Default: single-scale only. Derived classes should override for proper pyramid.
        return [EstimateFlow(frame0, frame1)];
    }

    /// <summary>
    /// Computes forward-backward consistency between two flow fields.
    /// </summary>
    /// <param name="forwardFlow">Forward flow from frame0 to frame1 [2, height, width].</param>
    /// <param name="backwardFlow">Backward flow from frame1 to frame0 [2, height, width].</param>
    /// <returns>Consistency map [height, width] where low values indicate consistent flow and high values indicate occlusion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If we track a pixel from frame0 to frame1 (forward flow) and then
    /// back from frame1 to frame0 (backward flow), it should end up at the same position.
    /// The consistency check measures how far off this round-trip is. Large errors indicate
    /// that the pixel is occluded (hidden) in one of the frames.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> ComputeForwardBackwardConsistency(Tensor<T> forwardFlow, Tensor<T> backwardFlow)
    {
        int height = forwardFlow.Shape[1];
        int width = forwardFlow.Shape[2];

        var consistency = new Tensor<T>([height, width]);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double fwdDx = NumOps.ToDouble(forwardFlow.Data.Span[h * width + w]);
                double fwdDy = NumOps.ToDouble(forwardFlow.Data.Span[height * width + h * width + w]);

                // Target position in frame1
                int tgtX = Math.Max(0, Math.Min((int)Math.Round(w + fwdDx), width - 1));
                int tgtY = Math.Max(0, Math.Min((int)Math.Round(h + fwdDy), height - 1));

                // Backward flow at target position
                double bwdDx = NumOps.ToDouble(backwardFlow.Data.Span[tgtY * width + tgtX]);
                double bwdDy = NumOps.ToDouble(backwardFlow.Data.Span[height * width + tgtY * width + tgtX]);

                // Round-trip error
                double errX = fwdDx + bwdDx;
                double errY = fwdDy + bwdDy;
                double error = Math.Sqrt(errX * errX + errY * errY);

                consistency.Data.Span[h * width + w] = NumOps.FromDouble(error);
            }
        }

        return consistency;
    }

    /// <summary>
    /// Computes the endpoint error (EPE) between estimated and ground truth flow.
    /// </summary>
    /// <param name="estimatedFlow">Estimated flow [2, height, width].</param>
    /// <param name="groundTruthFlow">Ground truth flow [2, height, width].</param>
    /// <returns>Mean endpoint error (scalar).</returns>
    public T ComputeEndpointError(Tensor<T> estimatedFlow, Tensor<T> groundTruthFlow)
    {
        int height = estimatedFlow.Shape[1];
        int width = estimatedFlow.Shape[2];
        double totalError = 0;
        int count = height * width;

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double estDx = NumOps.ToDouble(estimatedFlow.Data.Span[h * width + w]);
                double estDy = NumOps.ToDouble(estimatedFlow.Data.Span[height * width + h * width + w]);
                double gtDx = NumOps.ToDouble(groundTruthFlow.Data.Span[h * width + w]);
                double gtDy = NumOps.ToDouble(groundTruthFlow.Data.Span[height * width + h * width + w]);

                double errX = estDx - gtDx;
                double errY = estDy - gtDy;
                totalError += Math.Sqrt(errX * errX + errY * errY);
            }
        }

        return NumOps.FromDouble(totalError / count);
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For optical flow, input should contain two frames stacked
        // Split and estimate flow
        int channels = input.Shape[1] / 2;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var frame0 = new Tensor<T>([channels, height, width]);
        var frame1 = new Tensor<T>([channels, height, width]);

        int halfSize = channels * height * width;
        for (int i = 0; i < halfSize; i++)
        {
            frame0.Data.Span[i] = input.Data.Span[i];
            frame1.Data.Span[i] = input.Data.Span[halfSize + i];
        }

        return EstimateFlow(frame0, frame1);
    }
}
