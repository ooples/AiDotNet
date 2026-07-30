using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using System.IO;

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
    /// Gets whether <see cref="EstimateFlow"/> accepts a BATCHED frame pair
    /// (<c>[batch, channels, height, width]</c>) and returns <c>[batch, 2, height, width]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Defaults to <c>false</c>, matching the documented rank-3 contract above. When a derived model
    /// genuinely handles a batch dimension, overriding this to <c>true</c> lets
    /// <see cref="PredictCore"/> estimate the whole batch in ONE forward pass instead of looping over
    /// samples — the loop runs the entire flow network once per sample, which dominates the cost of a
    /// batched prediction.
    /// </para>
    /// <para>
    /// Only override this after confirming the implementation really is batch-correct. Claiming batch
    /// support that does not exist produces silently wrong flow rather than an error, because a
    /// rank-3-only implementation will happily interpret the batch axis as channels.
    /// </para>
    /// </remarks>
    protected virtual bool SupportsBatchedEstimateFlow => false;

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
        if (forwardFlow.Rank < 3 || forwardFlow.Shape[0] < 2)
            throw new ArgumentException("Forward flow must have shape [2, height, width].", nameof(forwardFlow));
        if (backwardFlow.Rank < 3 || backwardFlow.Shape[0] < 2)
            throw new ArgumentException("Backward flow must have shape [2, height, width].", nameof(backwardFlow));
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
        if (estimatedFlow.Rank < 3 || estimatedFlow.Shape[0] < 2)
            throw new ArgumentException("Estimated flow must have shape [2, height, width].", nameof(estimatedFlow));
        if (groundTruthFlow.Rank < 3 || groundTruthFlow.Shape[0] < 2)
            throw new ArgumentException("Ground truth flow must have shape [2, height, width].", nameof(groundTruthFlow));
        if (estimatedFlow.Shape[1] != groundTruthFlow.Shape[1] || estimatedFlow.Shape[2] != groundTruthFlow.Shape[2])
            throw new ArgumentException("Estimated and ground truth flows must have the same spatial dimensions.");
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // For optical flow, input should contain two frames stacked [batch, 2*channels, height, width]
        if (input.Rank < 4)
            throw new ArgumentException($"Input must be rank 4 [batch, 2*channels, height, width], got rank {input.Rank}.", nameof(input));
        if (input.Shape[1] % 2 != 0)
            throw new ArgumentException($"Input channel dimension must be even (two frames stacked), got {input.Shape[1]}.", nameof(input));
        int batch = input.Shape[0];
        int channels = input.Shape[1] / 2;
        int height = input.Shape[2];
        int width = input.Shape[3];

        // Split the stacked pair along the CHANNEL axis with recorded narrows.
        //
        // This previously allocated two fresh tensors and copied element-by-element through
        // Data.Span, which had three defects. (1) A raw buffer write is not a recorded operation, so
        // the frames arrived as tape leaves and no gradient could reach the caller's input — which is
        // precisely what an optical-flow term in a loss needs, since it differentiates the flow with
        // respect to the frames. (2) It indexed the flat buffer as [0, halfSize) and
        // [halfSize, 2*halfSize), which is only the correct channel split when batch == 1; for any
        // larger batch it silently mixed sample 0's channels with sample 1's data. (3) It allocated
        // per call and looped per element where one narrow suffices.
        var frames0 = Engine.TensorNarrow(input, 1, 0, channels);        // [B, C, H, W]
        var frames1 = Engine.TensorNarrow(input, 1, channels, channels); // [B, C, H, W]

        if (batch > 1 && SupportsBatchedEstimateFlow)
        {
            // One forward pass for the whole batch. Estimating per sample would run the entire flow
            // network `batch` times, which is the dominant cost of this method — enough to push
            // training tests over their time budget under parallel load.
            var batchedFlow = EstimateFlow(frames0, frames1);
            return batchedFlow.Rank == 3
                ? Engine.Reshape(
                    batchedFlow,
                    [1, batchedFlow.Shape[0], batchedFlow.Shape[1], batchedFlow.Shape[2]])
                : batchedFlow;
        }

        if (batch > 1)
        {
            // EstimateFlow is defined for a single frame pair, so estimate per sample and stack the
            // results rather than conflating samples. Correct but `batch` times the work; models that
            // accept a batched pair should opt into the fast path above.
            var perSample = new Tensor<T>[batch];
            for (int b = 0; b < batch; b++)
            {
                var f0 = Engine.Reshape(
                    Engine.TensorNarrow(frames0, 0, b, 1), [channels, height, width]);
                var f1 = Engine.Reshape(
                    Engine.TensorNarrow(frames1, 0, b, 1), [channels, height, width]);
                var sampleFlow = EstimateFlow(f0, f1);
                perSample[b] = sampleFlow.Rank == 3
                    ? Engine.Reshape(
                        sampleFlow,
                        [1, sampleFlow.Shape[0], sampleFlow.Shape[1], sampleFlow.Shape[2]])
                    : sampleFlow;
            }

            return Engine.Concat(perSample, 0);
        }

        var frame0 = Engine.Reshape(frames0, [channels, height, width]);
        var frame1 = Engine.Reshape(frames1, [channels, height, width]);

        // EstimateFlow returns rank-3 [2, H, W] (single frame-pair flow field
        // per the public API contract). Promote to rank-4 [B, 2, H, W] so
        // the Predict output matches the model's training-time forward shape
        // — the test scaffold's CreateRandomTargetTensor (sized via
        // EffectiveOutputShape inferred from Predict) needs to align with
        // what the base class's TrainWithTape feeds into the loss function,
        // and the framework convention is batched tensors everywhere.
        // Without this, Predict reports [2, H, W] while ForwardForTraining
        // emits [B, 2, H, W]; MSE then computes a degenerate flat-span loss
        // whose gradient direction stays constant across iterations and the
        // memorization invariant flags the model as "loss not decreasing".
        var flow = EstimateFlow(frame0, frame1);
        if (flow.Rank == 3 && batch == 1)
        {
            return Engine.Reshape(flow, [1, flow.Shape[0], flow.Shape[1], flow.Shape[2]]);
        }
        return flow;
    }

    /// <summary>
    /// Re-links the typed optical-flow role layers to the layers the base already deserialized
    /// (trained, shape-resolved) instead of re-running InitializeLayers, which would allocate fresh
    /// random-init convolutions and leave the trained weights unused in <c>Layers</c> (the #1221-class
    /// "clone/load predicts from random init" bug). Shared by the flow models whose serialized layout
    /// is identical: <c>[featureExtract, ...processingBlocks (numLayers), outputConv]</c>.
    /// </summary>
    /// <param name="numLayers">Deserialized processing-block count (untrusted — validated here).</param>
    /// <param name="modelName">Model name, used only in diagnostic messages.</param>
    /// <param name="featureExtract">Receives the resolved feature-extractor convolution.</param>
    /// <param name="processingBlocks">Caller-owned list; cleared and refilled with the resolved blocks.</param>
    /// <param name="outputConv">Receives the resolved output convolution.</param>
    protected void RelinkOpticalFlowLayers(
        int numLayers,
        string modelName,
        out ConvolutionalLayer<T> featureExtract,
        List<ConvolutionalLayer<T>> processingBlocks,
        out ConvolutionalLayer<T> outputConv)
    {
        // Validate the untrusted deserialized count first: a negative value would slip past the size
        // check below (Layers.Count < negative is false) and then throw a raw IndexOutOfRangeException
        // at Layers[numLayers + 1]. The ctor already enforces numLayers > 0.
        if (numLayers < 0)
            throw new InvalidDataException($"{modelName} serialized layer count {numLayers} is invalid.");
        // Compare in long space: numLayers + 2 as unchecked int wraps negative for numLayers near
        // int.MaxValue, bypassing this guard and throwing a raw IndexOutOfRangeException at
        // Layers[numLayers + 1] below. (long)numLayers + 2 cannot overflow for any int input.
        if (Layers.Count < (long)numLayers + 2)
            throw new InvalidDataException(
                $"{modelName} serialized layer count {Layers.Count} is too small for {numLayers} processing blocks.");
        featureExtract = Layers[0] as ConvolutionalLayer<T>
            ?? throw new InvalidDataException($"{modelName} feature extractor layer is missing or has the wrong type.");
        processingBlocks.Clear();
        for (int i = 0; i < numLayers; i++)
        {
            processingBlocks.Add(Layers[i + 1] as ConvolutionalLayer<T>
                ?? throw new InvalidDataException($"{modelName} processing block {i} is missing or has the wrong type."));
        }
        outputConv = Layers[numLayers + 1] as ConvolutionalLayer<T>
            ?? throw new InvalidDataException($"{modelName} output layer is missing or has the wrong type.");
    }
}
