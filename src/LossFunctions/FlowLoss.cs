using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video;
using AiDotNet.Video.Motion;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Optical-flow consistency loss: penalizes a reconstruction whose MOTION differs from the ground
/// truth's motion, even where individual frames already match.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> comparing video frames one at a time cannot detect flicker — every frame can
/// be individually close to the target while the sequence still jitters. This loss measures the
/// apparent motion between consecutive frames (the "optical flow") for both the reconstruction and the
/// ground truth, and penalizes the difference. It therefore constrains how things MOVE, not just how
/// they look.
/// </para>
/// <para>
/// <b>Definition.</b> For each adjacent frame pair,
/// <c>|| OF(x_t^rec, x_{t-1}^rec) - OF(x_t^GT, x_{t-1}^GT) ||_2^2</c>, as specified by Stream-DiffVSR
/// (Shiu et al., 2025) for its temporal-decoder objective, where the paper states the flow is computed
/// "using RAFT". The ground-truth flow is a constant target, so it is detached from the tape; gradients
/// reach the reconstruction through the flow estimator's inputs.
/// </para>
/// <para>
/// <b>The estimator is frozen.</b> The flow network is a measuring instrument, not something being
/// trained: its parameters are never registered with the model being optimized, so an optimizer walking
/// the model's parameters cannot touch them. Any of the library's optical-flow models can be
/// substituted; <see cref="RAFT{T}"/> is the default because it is what the paper specifies.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// // Paper default (RAFT).
/// var flowLoss = new FlowLoss&lt;float&gt;();
/// var term = flowLoss.ComputeTapeLoss(reconstructedClip, groundTruthClip);
///
/// // Substitute a different estimator.
/// var cheap = new FlowLoss&lt;float&gt;(new NeuFlowV2&lt;float&gt;());
/// </code>
/// </para>
/// </remarks>
public class FlowLoss<T> : LossFunctionBase<T>
{
    #region Fields

    private readonly OpticalFlowBase<T> _flowEstimator;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates the flow-consistency loss.
    /// </summary>
    /// <param name="flowEstimator">The optical-flow model used to measure motion. Defaults to
    /// <see cref="RAFT{T}"/>, the estimator Stream-DiffVSR specifies for this term.</param>
    public FlowLoss(OpticalFlowBase<T>? flowEstimator = null)
    {
        _flowEstimator = flowEstimator ?? new RAFT<T>();
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the optical-flow model used to measure motion. Exposed so callers can confirm which
    /// estimator a configured loss is using, and so it can be put in evaluation mode.
    /// </summary>
    public OpticalFlowBase<T> FlowEstimator => _flowEstimator;

    #endregion

    #region Tape Loss

    /// <inheritdoc/>
    /// <param name="predicted">Reconstructed clip, rank-4 <c>[frames, channels, height, width]</c> or
    /// rank-5 <c>[batch, frames, channels, height, width]</c>. Any batch size is accepted.</param>
    /// <param name="target">Ground-truth clip with the same shape.</param>
    /// <remarks>
    /// <para>
    /// Averages the squared flow difference over every adjacent frame pair. The squared L2 norm is
    /// normalized per element rather than summed so the term's magnitude does not scale with patch
    /// size — the paper's weight for this term was tuned at a fixed 512x512 patch size, so a
    /// resolution-dependent magnitude would silently re-weight the objective at any other size.
    /// </para>
    /// <para>
    /// A rank-5 clip is averaged over the batch as well, one sample at a time. Optical flow is
    /// defined per sample, so the samples cannot be folded together into one estimate; the loop is
    /// what makes the advertised rank-5 support real rather than rank-5-if-batch-is-1.
    /// </para>
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));

        var ps = predicted.Shape;
        var ts = target.Shape;
        if (ps.Length != ts.Length)
        {
            throw new ArgumentException(
                $"FlowLoss requires matching ranks; got predicted rank {ps.Length} and target rank {ts.Length}.",
                nameof(target));
        }

        if (ps.Length is not (4 or 5))
        {
            throw new ArgumentException(
                "FlowLoss requires a frame SEQUENCE: rank-4 [frames, channels, height, width] or rank-5 " +
                $"[batch, frames, channels, height, width]. Got rank {ps.Length}. A single frame carries no " +
                "motion, so this term is undefined for it.",
                nameof(predicted));
        }

        for (int d = 0; d < ps.Length; d++)
        {
            if (ps[d] != ts[d])
            {
                throw new ArgumentException(
                    $"FlowLoss requires identical shapes; they differ at dimension {d} " +
                    $"({ps[d]} vs {ts[d]}).",
                    nameof(target));
            }
        }

        // Frame axis is 0 for rank-4 [T,C,H,W] and 1 for rank-5 [B,T,C,H,W].
        int frameAxis = ps.Length == 5 ? 1 : 0;
        int numFrames = ps[frameAxis];
        if (numFrames < 2)
        {
            throw new ArgumentException(
                $"FlowLoss needs at least 2 frames to measure motion; got {numFrames}.",
                nameof(predicted));
        }

        if (ps.Length == 4) return ClipLoss(predicted, target, frameAxis, numFrames);

        // Rank-5: one sample at a time, then average. Flow is defined per sample, so the batch cannot
        // be folded into another axis; it has to be walked.
        int batch = ps[0];
        if (batch < 1)
        {
            throw new ArgumentException(
                $"FlowLoss received a rank-5 clip with batch {batch}. The per-dimension check above only " +
                "requires the two clips to agree, so an empty batch reaches here as a shape both sides " +
                "share; there is nothing to average over.",
                nameof(predicted));
        }

        Tensor<T> batchTotal = SampleLoss(predicted, target, frameAxis, numFrames, 0);
        for (int b = 1; b < batch; b++)
        {
            batchTotal = Engine.TensorAdd(batchTotal, SampleLoss(predicted, target, frameAxis, numFrames, b));
        }

        return Engine.TensorMultiplyScalar(batchTotal, NumOps.FromDouble(1.0 / batch));
    }

    /// <summary>Narrows both clips to sample <paramref name="sample"/> and measures that sample alone.</summary>
    private Tensor<T> SampleLoss(Tensor<T> predicted, Tensor<T> target, int frameAxis, int numFrames, int sample)
    {
        var onePredicted = Engine.TensorNarrow(predicted, 0, sample, 1);
        var oneTarget = Engine.TensorNarrow(target, 0, sample, 1);

        return ClipLoss(onePredicted, oneTarget, frameAxis, numFrames);
    }

    /// <summary>
    /// The mean squared flow difference over every adjacent frame pair of a single clip.
    /// </summary>
    /// <remarks>
    /// Split out from <see cref="ComputeTapeLoss"/> so the rank-4 path and each sample of the rank-5
    /// path run exactly the same computation. Two copies of a frame loop is how the two ranks drift
    /// into meaning different things.
    /// </remarks>
    private Tensor<T> ClipLoss(Tensor<T> predicted, Tensor<T> target, int frameAxis, int numFrames)
    {
        // numFrames >= 2 was enforced by the caller, so there is at least one pair and the
        // accumulator can start from a real value rather than from null.
        Tensor<T> total = PairLoss(predicted, target, frameAxis, 1);
        for (int t = 2; t < numFrames; t++)
        {
            total = Engine.TensorAdd(total, PairLoss(predicted, target, frameAxis, t));
        }

        return Engine.TensorMultiplyScalar(total, NumOps.FromDouble(1.0 / (numFrames - 1)));
    }

    /// <summary>The squared flow difference between frame <paramref name="t"/> and the one before it.</summary>
    private Tensor<T> PairLoss(Tensor<T> predicted, Tensor<T> target, int frameAxis, int t)
    {
        var predCurr = FrameAt(predicted, frameAxis, t);
        var predPrev = FrameAt(predicted, frameAxis, t - 1);
        var gtCurr = FrameAt(target, frameAxis, t);
        var gtPrev = FrameAt(target, frameAxis, t - 1);

        var predFlow = _flowEstimator.EstimateFlow(predCurr, predPrev);

        // The ground-truth flow is a fixed target: detaching it keeps the gradient path solely
        // through the reconstruction, and stops the estimator being nudged to explain the GT.
        var gtFlow = Engine.StopGradient(_flowEstimator.EstimateFlow(gtCurr, gtPrev));

        var diff = Engine.TensorSubtract(predFlow, gtFlow);
        var sq = Engine.TensorMultiply(diff, diff);
        var axes = Enumerable.Range(0, sq.Shape.Length).ToArray();

        return Engine.ReduceMean(sq, axes, keepDims: false);
    }

    /// <summary>
    /// Extracts frame <paramref name="index"/> as a <c>[channels, height, width]</c> tensor, keeping the
    /// slice on the autodiff tape.
    /// </summary>
    /// <remarks>
    /// Uses <c>Engine.TensorNarrow</c> rather than a bare tensor slice view: a raw view is not a
    /// recorded operation, so gradients would stop at the slice and the reconstruction would receive
    /// none of this term's signal.
    /// </remarks>
    private Tensor<T> FrameAt(Tensor<T> clip, int frameAxis, int index)
    {
        var frame = Engine.TensorNarrow(clip, frameAxis, index, 1);

        // Drop the now-singleton frame axis (and the batch axis for rank-5) so the estimator receives
        // the [C,H,W] frame its contract specifies.
        var shape = frame.Shape;
        var collapsed = new int[3];
        int write = 0;
        for (int d = 0; d < shape.Length; d++)
        {
            if (d == frameAxis) continue;
            if (frameAxis == 1 && d == 0)
            {
                // Rank-5: folding the batch into the channel axis is NOT valid, so this drops a
                // singleton batch axis and nothing else. ComputeTapeLoss narrows to one sample before
                // calling in, so a larger batch arriving here means that loop was bypassed -- an
                // internal invariant rather than a caller error.
                if (shape[0] != 1)
                {
                    throw new NotSupportedException(
                        $"FlowLoss received a rank-5 clip with batch {shape[0]}. Optical-flow estimation " +
                        "is defined per sample; pass one sample at a time so motion from different " +
                        "samples is not conflated.");
                }

                continue;
            }

            collapsed[write++] = shape[d];
        }

        return Engine.Reshape(frame, collapsed);
    }

    #endregion

    #region Unsupported Flat-Vector API

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">
    /// Always thrown: a flat vector carries no frame/channel/spatial structure, and motion cannot be
    /// recovered from it. Use <see cref="ComputeTapeLoss"/> with a shaped clip.
    /// </exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "FlowLoss operates on frame sequences and cannot infer frame boundaries from a flat " +
            "vector. Use ComputeTapeLoss(Tensor, Tensor) with a rank-4 or rank-5 clip.");
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">
    /// Always thrown, for the same reason as <see cref="CalculateLoss"/>. Gradients come from the
    /// autodiff tape via <see cref="ComputeTapeLoss"/>.
    /// </exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "FlowLoss operates on frame sequences and cannot infer frame boundaries from a flat " +
            "vector. Its gradient is produced by the autodiff tape through ComputeTapeLoss.");
    }

    #endregion
}
