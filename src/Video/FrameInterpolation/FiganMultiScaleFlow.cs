using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// FIGAN's multi-scale residual flow estimator and its occlusion-aware bidirectional synthesis.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// van Amersfoort et al., arXiv:1711.06045. The paper estimates flow coarse-to-fine over
/// <c>J = 3</c> scales, refining the upsampled coarser estimate at each finer level:
/// </para>
/// <code>
///   Gamma_j = f_flow_coarse(D^j I_0, D^j I_1)                    if j = J
///           = f_flow_refine(D^j I_0, D^j I_1, U Gamma_(j+1))      otherwise
///
///   f_flow_refine(I_0, I_1, Gamma) = tanh(Gamma + Gamma_res)
///   Gamma_res = f_flow_res(I_0(-Delta), I_1(Delta), Gamma)
///
///   I_0.5^syn = W o I_0(-Delta) + (1 - W) o I_1(Delta)
/// </code>
/// <para>
/// <b>Gamma carries THREE channels, not two.</b> Table 1 gives every flow module <c>N_o = 3</c>, and
/// the synthesis needs a per-pixel occlusion weight <c>W</c> alongside the two displacement
/// components — so <c>Gamma = (dx, dy, W)</c>. The residual module's <c>N_i = 9</c> confirms it: the
/// two warped RGB frames (3 + 3) plus the current 3-channel Gamma. Reading Gamma as pure 2-channel
/// flow leaves the input widths unexplainable and W with nowhere to come from.
/// </para>
/// <para>
/// <b>The tanh is on the SUM, not the residual.</b> <c>tanh(Gamma + Gamma_res)</c> bounds the refined
/// estimate; applying it to the residual alone would leave the accumulated flow unbounded across
/// scales, which is what the bound exists to prevent.
/// </para>
/// <para>
/// <b>Both directions are warped and blended by a LEARNED W.</b> The mid-frame is not an average:
/// <c>I_0</c> is warped forward by <c>-Delta</c>, <c>I_1</c> backward by <c>+Delta</c>, and W decides
/// per pixel which to trust — that is how occlusions are handled. A fixed 0.5 blend is the naive
/// baseline the paper improves on.
/// </para>
/// <para><b>For Beginners:</b> To invent a frame between two others you first guess how things moved,
/// starting from a blurry small version and sharpening the guess step by step. Then you drag both
/// frames toward the middle and, pixel by pixel, decide which one to believe — because things hidden
/// in one frame may be visible in the other.</para>
/// </remarks>
public sealed class FiganMultiScaleFlow<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Channels in Gamma: two displacement components plus the occlusion weight.</summary>
    public const int FlowChannels = 3;

    private readonly int _scales;

    /// <summary>Gets the number of scales J. The paper uses 3.</summary>
    public int Scales => _scales;

    /// <summary>
    /// Creates the estimator.
    /// </summary>
    /// <param name="scales">Number of scales J. The paper uses 3.</param>
    public FiganMultiScaleFlow(int scales = 3)
    {
        if (scales <= 0)
            throw new ArgumentOutOfRangeException(nameof(scales), scales,
                "At least one scale is required; the paper uses J = 3.");
        _scales = scales;
    }

    /// <summary>
    /// Downsamples by successive halving, <c>D^level</c>.
    /// </summary>
    /// <param name="frame">A <c>[height, width, channels]</c> frame.</param>
    /// <param name="level">How many halvings to apply. Zero returns the input unchanged.</param>
    /// <remarks>
    /// Box-average halving. Each level averages 2x2 blocks, so a coarse level genuinely summarises its
    /// finer one rather than sub-sampling it — subsampling would alias motion and give the coarse flow
    /// estimate something different to fit than the fine one refines.
    /// </remarks>
    public Tensor<T> Downsample(Tensor<T> frame, int level)
    {
        if (frame is null) throw new ArgumentNullException(nameof(frame));
        if (level < 0) throw new ArgumentOutOfRangeException(nameof(level), level, "Level cannot be negative.");

        var current = frame;
        for (int l = 0; l < level; l++)
        {
            int h = current.Shape[0], w = current.Shape[1], c = current.Shape[2];
            int nh = Math.Max(1, h / 2), nw = Math.Max(1, w / 2);
            var next = new Tensor<T>(new[] { nh, nw, c });

            for (int y = 0; y < nh; y++)
            {
                for (int x = 0; x < nw; x++)
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        double sum = 0.0;
                        int count = 0;
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int sy = (y * 2) + dy, sx = (x * 2) + dx;
                                if (sy >= h || sx >= w) continue;
                                sum += Ops.ToDouble(current[((sy * w) + sx) * c + ch]);
                                count++;
                            }
                        }
                        next[((y * nw) + x) * c + ch] = Ops.FromDouble(sum / Math.Max(1, count));
                    }
                }
            }
            current = next;
        }
        return current;
    }

    /// <summary>
    /// Upsamples a flow field by 2x, <c>U</c>, scaling the displacement channels with it.
    /// </summary>
    /// <remarks>
    /// The displacements are MULTIPLIED by two as the resolution doubles: a motion of one pixel at a
    /// coarse level is two pixels at the next. Omitting that scaling is a silent halving of motion at
    /// every level and is the classic error in coarse-to-fine flow. The occlusion channel is a weight,
    /// not a displacement, so it is interpolated WITHOUT scaling.
    /// </remarks>
    public Tensor<T> UpsampleFlow(Tensor<T> flow, int targetHeight, int targetWidth)
    {
        if (flow is null) throw new ArgumentNullException(nameof(flow));
        if (flow.Shape[2] != FlowChannels)
            throw new ArgumentException(
                $"Gamma must have {FlowChannels} channels (dx, dy, W); got {flow.Shape[2]}.", nameof(flow));

        int h = flow.Shape[0], w = flow.Shape[1];
        double scaleY = (double)targetHeight / h;
        double scaleX = (double)targetWidth / w;
        var result = new Tensor<T>(new[] { targetHeight, targetWidth, FlowChannels });

        for (int y = 0; y < targetHeight; y++)
        {
            int sy = Math.Min(h - 1, (int)(y / scaleY));
            for (int x = 0; x < targetWidth; x++)
            {
                int sx = Math.Min(w - 1, (int)(x / scaleX));
                int src = ((sy * w) + sx) * FlowChannels;
                int dst = ((y * targetWidth) + x) * FlowChannels;

                // Displacements scale with resolution; the occlusion weight does not.
                result[dst] = Ops.FromDouble(Ops.ToDouble(flow[src]) * scaleX);
                result[dst + 1] = Ops.FromDouble(Ops.ToDouble(flow[src + 1]) * scaleY);
                result[dst + 2] = flow[src + 2];
            }
        }
        return result;
    }

    /// <summary>
    /// Warps <paramref name="frame"/> by <paramref name="direction"/> times the flow, with bilinear
    /// sampling.
    /// </summary>
    /// <param name="frame">A <c>[height, width, channels]</c> frame.</param>
    /// <param name="flow">Gamma, <c>[height, width, 3]</c>.</param>
    /// <param name="direction">
    /// <c>-1</c> warps <c>I_0</c> forward to the midpoint, <c>+1</c> warps <c>I_1</c> backward. The
    /// paper's <c>I_0(-Delta)</c> and <c>I_1(Delta)</c>.
    /// </param>
    /// <remarks>
    /// Samples are clamped at the border rather than zero-filled: a zero fill injects black at every
    /// pixel whose source falls outside the frame, which the occlusion map then has to learn to mask
    /// around.
    /// </remarks>
    public Tensor<T> Warp(Tensor<T> frame, Tensor<T> flow, int direction)
    {
        if (frame is null) throw new ArgumentNullException(nameof(frame));
        if (flow is null) throw new ArgumentNullException(nameof(flow));

        int h = frame.Shape[0], w = frame.Shape[1], c = frame.Shape[2];
        if (flow.Shape[0] != h || flow.Shape[1] != w)
            throw new ArgumentException(
                $"Flow is [{flow.Shape[0]}, {flow.Shape[1]}] but the frame is [{h}, {w}]; they must " +
                "match, so upsample the flow before warping.", nameof(flow));

        var result = new Tensor<T>(new[] { h, w, c });

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int fi = ((y * w) + x) * FlowChannels;
                double sx = x + (direction * Ops.ToDouble(flow[fi]));
                double sy = y + (direction * Ops.ToDouble(flow[fi + 1]));

                // Clamp to the border.
                sx = Math.Min(Math.Max(sx, 0.0), w - 1.0);
                sy = Math.Min(Math.Max(sy, 0.0), h - 1.0);

                int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                int x1 = Math.Min(x0 + 1, w - 1), y1 = Math.Min(y0 + 1, h - 1);
                double ax = sx - x0, ay = sy - y0;

                for (int ch = 0; ch < c; ch++)
                {
                    double v00 = Ops.ToDouble(frame[((y0 * w) + x0) * c + ch]);
                    double v01 = Ops.ToDouble(frame[((y0 * w) + x1) * c + ch]);
                    double v10 = Ops.ToDouble(frame[((y1 * w) + x0) * c + ch]);
                    double v11 = Ops.ToDouble(frame[((y1 * w) + x1) * c + ch]);

                    double top = (v00 * (1 - ax)) + (v01 * ax);
                    double bottom = (v10 * (1 - ax)) + (v11 * ax);
                    result[((y * w) + x) * c + ch] = Ops.FromDouble((top * (1 - ay)) + (bottom * ay));
                }
            }
        }
        return result;
    }

    /// <summary>
    /// The paper's synthesis: <c>W o I_0(-Delta) + (1 - W) o I_1(Delta)</c>.
    /// </summary>
    /// <param name="frame0">The earlier frame.</param>
    /// <param name="frame1">The later frame.</param>
    /// <param name="flow">Gamma, whose third channel supplies W.</param>
    /// <remarks>
    /// W is squashed into <c>[0, 1]</c> by a logistic, so the third channel can be an unbounded
    /// activation while the blend stays a convex combination. Letting W leave <c>[0, 1]</c> would
    /// extrapolate beyond both frames and can produce values outside the valid intensity range.
    /// </remarks>
    public Tensor<T> Synthesise(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> flow)
    {
        var warped0 = Warp(frame0, flow, direction: -1);
        var warped1 = Warp(frame1, flow, direction: +1);

        int h = frame0.Shape[0], w = frame0.Shape[1], c = frame0.Shape[2];
        var result = new Tensor<T>(new[] { h, w, c });

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double raw = Ops.ToDouble(flow[(((y * w) + x) * FlowChannels) + 2]);
                double weight = 1.0 / (1.0 + Math.Exp(-raw));

                for (int ch = 0; ch < c; ch++)
                {
                    int i = ((y * w) + x) * c + ch;
                    result[i] = Ops.FromDouble(
                        (weight * Ops.ToDouble(warped0[i])) + ((1.0 - weight) * Ops.ToDouble(warped1[i])));
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Applies the paper's refinement to an upsampled coarser estimate:
    /// <c>tanh(Gamma + Gamma_res)</c>.
    /// </summary>
    /// <param name="upsampledFlow">U Gamma_(j+1).</param>
    /// <param name="residual">Gamma_res, from the residual module.</param>
    public Tensor<T> Refine(Tensor<T> upsampledFlow, Tensor<T> residual)
    {
        if (upsampledFlow is null) throw new ArgumentNullException(nameof(upsampledFlow));
        if (residual is null) throw new ArgumentNullException(nameof(residual));
        if (residual.Length != upsampledFlow.Length)
            throw new ArgumentException(
                $"Residual length {residual.Length} does not match the flow's {upsampledFlow.Length}.",
                nameof(residual));

        var result = new Tensor<T>(upsampledFlow.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
        {
            // tanh of the SUM — bounding the accumulated estimate, not just the correction.
            result[i] = Ops.FromDouble(Math.Tanh(
                Ops.ToDouble(upsampledFlow[i]) + Ops.ToDouble(residual[i])));
        }
        return result;
    }
}
