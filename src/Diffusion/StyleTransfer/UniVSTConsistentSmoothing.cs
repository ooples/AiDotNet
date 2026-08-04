using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// UniVST's sliding-window consistent smoothing (Song et al., arXiv:2410.20084, TPAMI 2025):
/// flow-warped temporal averaging in PIXEL space, fed back as refined noise.
/// </summary>
/// <remarks>
/// <para>
/// The distinguishing choice is that smoothing happens on decoded PIXELS and is then folded back
/// into the latent trajectory as a corrected noise estimate, rather than being applied to latents
/// directly. Averaging latents would blur structure, because neighbouring latents are not aligned;
/// averaging flow-warped pixels aligns them first.
/// </para>
/// <para>
/// The averaged frames are re-encoded and used to re-derive epsilon, so the correction re-enters the
/// diffusion process through the noise rather than overwriting the latent. That keeps the sampler's
/// own update rule intact.
/// </para>
/// <para><b>For Beginners:</b> Independently stylized frames flicker. This lines neighbouring frames
/// up using optical flow, averages them, and feeds the steadier result back into the generator.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class UniVSTConsistentSmoothing<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly UniVSTOptions _options;

    /// <summary>
    /// The ambient tensor engine. Read per use rather than captured in the constructor, so the
    /// smoother follows a later engine switch instead of pinning whichever engine happened to be
    /// current when it was built.
    /// </summary>
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Gets m, the window half-width; the full window spans 2m + 1 frames.</summary>
    public int HalfWindow => _options.SmoothingHalfWindow;

    /// <summary>Gets the full window length, 2m + 1.</summary>
    public int WindowLength => (2 * _options.SmoothingHalfWindow) + 1;

    /// <summary>Creates the smoother.</summary>
    /// <param name="options">Configuration; defaults are the paper's.</param>
    public UniVSTConsistentSmoothing(UniVSTOptions? options = null)
    {
        _options = options ?? new UniVSTOptions();
        _options.Validate();
    }

    /// <summary>True when smoothing applies at the given progress through the schedule (fraction of T).</summary>
    public bool IsActive(double timestepFraction) =>
        timestepFraction >= _options.SmoothingStartFraction &&
        timestepFraction <= _options.SmoothingEndFraction;

    /// <summary>
    /// Window indices for frame <paramref name="index"/>, clamped to the sequence.
    /// </summary>
    /// <remarks>
    /// Clamped rather than wrapped: frame 0 and frame N-1 have no true neighbours beyond the ends,
    /// and wrapping would average the last frame into the first and invent motion across a cut.
    /// </remarks>
    public IReadOnlyList<int> WindowIndices(int index, int frameCount)
    {
        if (frameCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameCount), frameCount, "frameCount must be positive.");
        if (index < 0 || index >= frameCount)
            throw new ArgumentOutOfRangeException(nameof(index), index, $"index must be in [0, {frameCount - 1}].");

        int m = _options.SmoothingHalfWindow;
        var result = new List<int>();
        for (int j = index - m; j <= index + m; j++)
        {
            if (j < 0 || j >= frameCount) continue;
            result.Add(j);
        }
        return result;
    }

    /// <summary>
    /// Smooths a decoded frame sequence: each frame becomes the mean of its window's neighbours,
    /// warped into its own frame of reference.
    /// </summary>
    /// <param name="frames">Decoded frames, each <c>[C,H,W]</c>.</param>
    /// <param name="flowTo">
    /// <c>flowTo[i][j]</c> is the flow field <c>[2,H,W]</c> that warps frame j into frame i's frame
    /// of reference. Bidirectional in the paper (RAFT), so both j &lt; i and j &gt; i are supplied.
    /// The diagonal entry may be null, since a frame needs no warping into itself.
    /// </param>
    /// <returns>The smoothed frames, one per input frame.</returns>
    public IReadOnlyList<Tensor<T>> SmoothSequence(
        IReadOnlyList<Tensor<T>> frames,
        Func<int, int, Tensor<T>?> flowTo)
    {
        if (frames == null) throw new ArgumentNullException(nameof(frames));
        if (flowTo == null) throw new ArgumentNullException(nameof(flowTo));
        if (frames.Count == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));

        var smoothed = new List<Tensor<T>>(frames.Count);

        for (int i = 0; i < frames.Count; i++)
        {
            var window = WindowIndices(i, frames.Count);
            var accumulator = new Tensor<T>(frames[i].Shape.ToArray());
            int contributed = 0;

            foreach (int j in window)
            {
                Tensor<T> aligned;
                if (j == i)
                {
                    aligned = frames[i];
                }
                else
                {
                    var flow = flowTo(i, j);
                    // No flow for this pair means no correspondence was established; skipping it is
                    // right, because warping by a zero field would average in an UNALIGNED frame and
                    // reintroduce exactly the ghosting this step removes.
                    if (flow == null) continue;
                    aligned = FlowWarpHelper.Warp(Engine, frames[j], flow);
                }

                for (int p = 0; p < accumulator.Length; p++)
                    accumulator[p] = NumOps.Add(accumulator[p], aligned[p]);
                contributed++;
            }

            // Divide by the number of frames ACTUALLY averaged, not the nominal 2m+1. At the
            // sequence ends the window is short, and dividing by 2m+1 there would darken those
            // frames toward zero.
            var mean = new Tensor<T>(frames[i].Shape.ToArray());
            if (contributed == 0)
            {
                for (int p = 0; p < mean.Length; p++) mean[p] = frames[i][p];
            }
            else
            {
                T divisor = NumOps.FromDouble(contributed);
                for (int p = 0; p < mean.Length; p++) mean[p] = NumOps.Divide(accumulator[p], divisor);
            }

            smoothed.Add(mean);
        }

        return smoothed;
    }

    /// <summary>
    /// Re-derives epsilon from a smoothed x0 prediction:
    /// <c>epsbar = (z_t - sqrt(alpha_t) * z0bar) / sqrt(1 - alpha_t)</c>.
    /// </summary>
    /// <param name="latentT">The current latent z_t.</param>
    /// <param name="smoothedZeroPrediction">The smoothed, re-encoded x0 prediction.</param>
    /// <param name="alphaCumulativeT">alpha-bar at t, in (0, 1].</param>
    public Tensor<T> RefineNoise(Tensor<T> latentT, Tensor<T> smoothedZeroPrediction, double alphaCumulativeT)
    {
        if (latentT == null) throw new ArgumentNullException(nameof(latentT));
        if (smoothedZeroPrediction == null) throw new ArgumentNullException(nameof(smoothedZeroPrediction));
        if (alphaCumulativeT is <= 0.0 or > 1.0)
            throw new ArgumentOutOfRangeException(nameof(alphaCumulativeT), alphaCumulativeT,
                "alphaCumulativeT must be in (0, 1].");
        if (latentT.Length != smoothedZeroPrediction.Length)
            throw new ArgumentException(
                $"Element counts must match: {latentT.Length} vs {smoothedZeroPrediction.Length}.",
                nameof(smoothedZeroPrediction));

        double sqrtAlpha = Math.Sqrt(alphaCumulativeT);
        double sqrtOneMinus = Math.Sqrt(1.0 - alphaCumulativeT);

        var result = new Tensor<T>(latentT.Shape.ToArray());

        // alpha-bar == 1 is t = 0: there is no noise component left to solve for, so the residual is
        // zero rather than a division by zero.
        if (sqrtOneMinus <= 0.0) return result;

        for (int i = 0; i < result.Length; i++)
        {
            double v = (NumOps.ToDouble(latentT[i]) - (sqrtAlpha * NumOps.ToDouble(smoothedZeroPrediction[i])))
                       / sqrtOneMinus;
            result[i] = NumOps.FromDouble(v);
        }
        return result;
    }

    /// <summary>
    /// The DDIM step rebuilt from the smoothed x0 and refined noise:
    /// <c>z_{t-1} = sqrt(alpha_{t-1}) * z0bar + sqrt(1 - alpha_{t-1}) * epsbar</c>.
    /// </summary>
    public Tensor<T> StepWithRefinedNoise(
        Tensor<T> smoothedZeroPrediction, Tensor<T> refinedNoise, double alphaCumulativePrevious)
    {
        if (smoothedZeroPrediction == null) throw new ArgumentNullException(nameof(smoothedZeroPrediction));
        if (refinedNoise == null) throw new ArgumentNullException(nameof(refinedNoise));
        if (alphaCumulativePrevious is < 0.0 or > 1.0)
            throw new ArgumentOutOfRangeException(nameof(alphaCumulativePrevious), alphaCumulativePrevious,
                "alphaCumulativePrevious must be in [0, 1].");
        if (smoothedZeroPrediction.Length != refinedNoise.Length)
            throw new ArgumentException(
                $"Element counts must match: {smoothedZeroPrediction.Length} vs {refinedNoise.Length}.",
                nameof(refinedNoise));

        double sqrtAlphaPrev = Math.Sqrt(alphaCumulativePrevious);
        double sqrtOneMinusPrev = Math.Sqrt(1.0 - alphaCumulativePrevious);

        var result = new Tensor<T>(smoothedZeroPrediction.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
        {
            double v = (sqrtAlphaPrev * NumOps.ToDouble(smoothedZeroPrediction[i]))
                       + (sqrtOneMinusPrev * NumOps.ToDouble(refinedNoise[i]));
            result[i] = NumOps.FromDouble(v);
        }
        return result;
    }
}
