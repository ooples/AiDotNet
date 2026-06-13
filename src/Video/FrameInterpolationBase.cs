using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for frame interpolation models that generate intermediate frames between existing frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Frame interpolation increases video frame rate by generating new frames between existing ones.
/// This base class provides:
///
/// - Arbitrary timestep interpolation (not just midpoint)
/// - Multi-frame interpolation (generate multiple intermediate frames)
/// - Flow-based and kernel-based interpolation utilities
/// - Temporal consistency support
///
/// Derived classes implement specific architectures like RIFE, AMT, EMA-VFI, VFIMamba, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Frame interpolation makes choppy video smoother by adding new frames
/// in between existing ones. For example, converting 30fps video to 60fps or even 120fps.
/// The model "imagines" what the scene looks like at the intermediate time points.
/// </para>
/// </remarks>
public abstract class FrameInterpolationBase<T> : VideoNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the temporal scale factor (e.g., 2 for doubling frame rate).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 2 means one intermediate frame is generated between each pair (30fps to 60fps).
    /// A value of 4 means three intermediate frames are generated (30fps to 120fps).
    /// </para>
    /// </remarks>
    private int _temporalScaleFactor = 2;

    public int TemporalScaleFactor
    {
        get => _temporalScaleFactor;
        protected set
        {
            if (value < 2)
                throw new ArgumentOutOfRangeException(nameof(value), value, "TemporalScaleFactor must be at least 2.");
            _temporalScaleFactor = value;
        }
    }

    /// <summary>
    /// Gets whether this model supports arbitrary timestep interpolation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model can interpolate at any time t in (0, 1) between two frames.
    /// When false, the model only supports fixed timesteps (e.g., midpoint t=0.5).
    /// </para>
    /// </remarks>
    public bool SupportsArbitraryTimestep { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the FrameInterpolationBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected FrameInterpolationBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Frame-interpolation models consume two RGB frames concatenated
    /// channel-wise — 2 × Architecture.InputDepth = 6 channels — but
    /// Architecture.InputDepth itself reports the SINGLE-FRAME count (3)
    /// so it matches the architecture's per-frame metadata. Returning null
    /// suppresses the base class's ResolveLazyLayerShapes pre-walk, which
    /// would size the first lazy ConvolutionalLayer for depth 3 and then
    /// every real Train()/Predict() with the [1, 6, H, W] concat would
    /// fail with "Expected input depth 3, but got 6". Same root-cause fix
    /// as UFM / RAPIDFlow (optical flow — the sibling 2-frame-concat
    /// family) and MisGAN / AutoDiffTabGenerator / GOGGLE / MGTSD. Layers
    /// resolve from the real concatenated input on first Forward instead.
    /// </summary>
    protected override int[]? TryGetArchitectureInputShape() => null;

    /// <summary>
    /// Interpolates a single intermediate frame between two input frames at the given timestep.
    /// </summary>
    /// <param name="frame0">First frame [channels, height, width].</param>
    /// <param name="frame1">Second frame [channels, height, width].</param>
    /// <param name="t">Timestep in (0, 1). t=0.5 is the midpoint.</param>
    /// <returns>Interpolated frame [channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given two consecutive video frames, this generates a new frame
    /// that shows what the scene looks like at time t between them.
    /// t=0.5 gives you the exact midpoint. t=0.25 gives a frame closer to frame0.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5);

    /// <summary>
    /// Interpolates multiple intermediate frames between two input frames.
    /// </summary>
    /// <param name="frame0">First frame [channels, height, width].</param>
    /// <param name="frame1">Second frame [channels, height, width].</param>
    /// <param name="numIntermediate">Number of intermediate frames to generate.</param>
    /// <returns>Tensor containing all intermediate frames [numIntermediate, channels, height, width].</returns>
    public virtual Tensor<T> InterpolateMulti(Tensor<T> frame0, Tensor<T> frame1, int numIntermediate)
    {
        if (frame0 is null) throw new ArgumentNullException(nameof(frame0));
        if (frame1 is null) throw new ArgumentNullException(nameof(frame1));
        if (frame0.Rank < 3)
            throw new ArgumentException("Frame must have at least 3 dimensions [channels, height, width].", nameof(frame0));
        if (frame1.Rank < 3)
            throw new ArgumentException("Frame must have at least 3 dimensions [channels, height, width].", nameof(frame1));
        if (numIntermediate < 1)
            throw new ArgumentOutOfRangeException(nameof(numIntermediate), "Must generate at least 1 intermediate frame.");

        int channels = frame0.Shape[0];
        int height = frame0.Shape[1];
        int width = frame0.Shape[2];

        var result = new Tensor<T>([numIntermediate, channels, height, width]);

        for (int i = 0; i < numIntermediate; i++)
        {
            double t = (i + 1.0) / (numIntermediate + 1.0);
            var interpolated = Interpolate(frame0, frame1, t);
            StoreFrame(result, interpolated, i);
        }

        return result;
    }

    /// <summary>
    /// Interpolates all frames in a video sequence to increase frame rate.
    /// </summary>
    /// <param name="frames">Input frames [numFrames, channels, height, width].</param>
    /// <returns>Interpolated sequence with (numFrames - 1) * scaleFactor + 1 frames.</returns>
    public virtual Tensor<T> InterpolateSequence(Tensor<T> frames)
    {
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        if (frames.Rank < 4)
            throw new ArgumentException("Input must have 4 dimensions [numFrames, channels, height, width].", nameof(frames));
        if (frames.Shape[0] < 2)
            throw new ArgumentException("At least 2 frames are required for interpolation.", nameof(frames));

        int numFrames = frames.Shape[0];
        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        int numIntermediate = TemporalScaleFactor - 1;
        int outputFrames = (numFrames - 1) * TemporalScaleFactor + 1;

        var result = new Tensor<T>([outputFrames, channels, height, width]);

        for (int i = 0; i < numFrames; i++)
        {
            var frame = ExtractFrame(frames, i);
            StoreFrame(result, frame, i * TemporalScaleFactor);

            if (i < numFrames - 1)
            {
                var nextFrame = ExtractFrame(frames, i + 1);
                for (int j = 1; j <= numIntermediate; j++)
                {
                    double t = (double)j / TemporalScaleFactor;
                    var interpolated = Interpolate(frame, nextFrame, t);
                    StoreFrame(result, interpolated, i * TemporalScaleFactor + j);
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Two-frame interpolation models accept either:
    /// <list type="bullet">
    ///   <item>A rank-4 frame sequence <c>[N, C, H, W]</c> — calls
    ///         <see cref="InterpolateSequence"/>. Rank-4 input is always treated
    ///         as a sequence; previously this routed even-channel sequences
    ///         (RGBA, 2-channel optical flow, etc.) into the pair-concat path
    ///         and split them by channel instead of frame.</item>
    ///   <item>A channel-concatenated frame pair <c>[2C, H, W]</c> emitted by the
    ///         standard test scaffold — splits and calls <see cref="Interpolate"/>
    ///         with t = 0.5. Batched channel-concatenated input <c>[B, 2C, H, W]</c>
    ///         is no longer auto-detected (it is indistinguishable from a frame
    ///         sequence with even C); callers needing batched pair-concat must
    ///         iterate the batch dimension and call <see cref="Interpolate"/>
    ///         directly.</item>
    /// </list>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        // Rank-4 always means a frame *sequence* [N, C, H, W]. Treating it as
        // a channel-concatenated pair purely on "C is even" misclassified
        // RGBA (C=4), depth-with-confidence (C=2), and any other multi-channel
        // sequence input. The unambiguous channel-concat shape is rank-3
        // [2C, H, W] — that's what we keep here.
        bool isPairConcat = input.Rank == 3 && input.Shape[0] % 2 == 0 && input.Shape[0] > 0;
        if (isPairConcat) return InterpolatePairConcat(input);

        // Common caller mistake: passing batched pair-concat [1, 2C, H, W]
        // expecting channel-split semantics. With rank-4 → InterpolateSequence,
        // that input would die in the sequence path with the unhelpful
        // "At least 2 frames are required" message. Detect and redirect.
        if (input.Rank == 4 && input.Shape[0] == 1 && input.Shape[1] % 2 == 0 && input.Shape[1] > 0)
        {
            throw new ArgumentException(
                $"Input shape [{input.Shape[0]}, {input.Shape[1]}, {input.Shape[2]}, {input.Shape[3]}] " +
                "looks like a batched channel-concatenated frame pair, but rank-4 input is treated " +
                "as a frame sequence [N, C, H, W] and the leading dim is 1, so InterpolateSequence " +
                "would reject it. If you intended a pair-concat, drop the batch dim and pass " +
                $"[{input.Shape[1]}, {input.Shape[2]}, {input.Shape[3]}] (rank-3), or iterate the " +
                "batch and call Interpolate(frame0, frame1) per pair. If this really is a " +
                "single-frame sequence, pad to at least 2 frames before calling Predict.",
                nameof(input));
        }

        return InterpolateSequence(input);
    }

    private Tensor<T> InterpolatePairConcat(Tensor<T> pair)
    {
        // Predict only routes rank-3 [2C, H, W] tensors here (see the rank-4
        // disambiguation in Predict). Rank-4 batched pair-concat is therefore
        // unreachable from the public surface; callers that need batched
        // pair-concat should iterate batches and call Interpolate directly.
        int channels = pair.Shape[0] / 2;
        int height = pair.Shape[1];
        int width = pair.Shape[2];
        int frameStride = channels * height * width;
        var src = pair.AsSpan();

        var frame0 = new Tensor<T>([channels, height, width]);
        var frame1 = new Tensor<T>([channels, height, width]);
        src.Slice(0, frameStride).CopyTo(frame0.AsWritableSpan());
        src.Slice(frameStride, frameStride).CopyTo(frame1.AsWritableSpan());
        return Interpolate(frame0, frame1, t: 0.5);
    }
}
