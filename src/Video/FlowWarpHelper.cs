using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video;

/// <summary>
/// Backward-warps an image by an optical-flow field, using tape-recorded engine operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> "warping" moves every pixel of an image along a motion field so it lines up
/// with a different frame. If you know how things moved between frame 1 and frame 2, warping frame 1
/// by that motion gives you a prediction of frame 2. Video models use this constantly: it is how the
/// previous frame's detail gets reused for the current frame instead of being re-invented (which is
/// what causes flicker).
/// </para>
/// <para>
/// <b>Why this exists.</b> Several models each carried their own copy of this operation written as a
/// nested per-pixel loop that wrote into a freshly allocated tensor. That has two consequences. First,
/// raw buffer writes are not recorded operations, so no gradient can flow back through a warp — any
/// loss term that compares warped content is then training nothing, and any model whose architecture
/// depends on warping is effectively a per-frame model. Second, a scalar loop over
/// batch x height x width x channels is orders of magnitude slower than the batched kernel underneath
/// <c>GridSample</c>.
/// </para>
/// <para>
/// <b>Method.</b> Flow is converted to a normalized sampling grid and handed to <c>GridSample</c>,
/// which is the same formulation as PyTorch's <c>grid_sample</c>: for output pixel <c>(x, y)</c> the
/// source location is <c>(x + flow_x, y + flow_y)</c>, mapped to <c>[-1, 1]</c>. The pixel coordinates
/// themselves are constants, so all gradient flows through the flow field and the source image —
/// exactly as required for a flow-based loss or a warped-conditioning path.
/// </para>
/// </remarks>
public static class FlowWarpHelper
{
    /// <summary>
    /// Cached constant coordinate grids, keyed by spatial size. The base pixel coordinates never
    /// change for a given resolution, so rebuilding them per frame would allocate on every call of
    /// what is typically an inner loop over a video.
    /// </summary>
    private static readonly ConcurrentDictionary<(int Height, int Width, Type Numeric), object> BaseGrids
        = new();

    /// <summary>
    /// Backward-warps <paramref name="image"/> by <paramref name="flow"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="engine">The tensor engine, so the operations are recorded on the caller's tape.
    /// Pass the calling type's own <c>Engine</c> member.</param>
    /// <param name="image">Image to warp, <c>[C,H,W]</c> or <c>[B,C,H,W]</c>.</param>
    /// <param name="flow">Flow field <c>[2,H,W]</c> or <c>[B,2,H,W]</c>, holding (dx, dy) in PIXELS.</param>
    /// <returns>The warped image, shaped like <paramref name="image"/>.</returns>
    public static Tensor<T> Warp<T>(IEngine engine, Tensor<T> image, Tensor<T> flow)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (flow is null) throw new ArgumentNullException(nameof(flow));

        var imgShape = image.Shape;
        var flowShape = flow.Shape;

        if (imgShape.Length is not (3 or 4))
        {
            throw new ArgumentException(
                $"Image must be [C,H,W] or [B,C,H,W]; got rank {imgShape.Length}.", nameof(image));
        }

        if (flowShape.Length is not (3 or 4))
        {
            throw new ArgumentException(
                $"Flow must be [2,H,W] or [B,2,H,W]; got rank {flowShape.Length}.", nameof(flow));
        }

        bool imageBatched = imgShape.Length == 4;
        int batch = imageBatched ? imgShape[0] : 1;
        int channels = imageBatched ? imgShape[1] : imgShape[0];
        int height = imgShape[imgShape.Length - 2];
        int width = imgShape[imgShape.Length - 1];

        int flowChannels = flowShape.Length == 4 ? flowShape[1] : flowShape[0];
        if (flowChannels < 2)
        {
            throw new ArgumentException(
                $"Flow must have at least 2 channels (dx, dy); got {flowChannels}.", nameof(flow));
        }

        int flowH = flowShape[flowShape.Length - 2];
        int flowW = flowShape[flowShape.Length - 1];
        if (flowH != height || flowW != width)
        {
            throw new ArgumentException(
                $"Flow spatial size [{flowH}, {flowW}] must match the image's [{height}, {width}]. " +
                "Resample the flow first (and scale its magnitudes accordingly).",
                nameof(flow));
        }

        // GridSample works in NCHW, so promote unbatched inputs to a single-sample batch.
        var imageNchw = imageBatched
            ? image
            : engine.Reshape(image, [1, channels, height, width]);
        var flowNchw = flowShape.Length == 4
            ? flow
            : engine.Reshape(flow, [1, flowChannels, height, width]);

        var numOps = MathHelper.GetNumericOperations<T>();

        // Split the flow into its dx and dy planes: [B, 1, H, W] each.
        var flowX = engine.TensorNarrow(flowNchw, 1, 0, 1);
        var flowY = engine.TensorNarrow(flowNchw, 1, 1, 1);

        var (baseX, baseY) = GetBaseGrids<T>(height, width, numOps);

        // Source location in pixels, then normalized to [-1, 1] as GridSample expects.
        // A degenerate axis (size 1) has no interior to interpolate across, so its normalized
        // coordinate is pinned to 0 (the centre) rather than dividing by zero.
        var srcX = engine.TensorAdd(flowX, baseX);
        var srcY = engine.TensorAdd(flowY, baseY);

        var normX = Normalize(engine, srcX, width, numOps);
        var normY = Normalize(engine, srcY, height, numOps);

        // [B, 1, H, W] -> [B, H, W, 1] is a pure reinterpretation: with a singleton channel axis the
        // element order is identical, so this is a recorded reshape rather than a data-moving permute.
        var gridX = engine.Reshape(normX, [batch, height, width, 1]);
        var gridY = engine.Reshape(normY, [batch, height, width, 1]);
        var grid = engine.Concat([gridX, gridY], 3);          // [B, H, W, 2]

        var warped = engine.GridSample(imageNchw, grid);      // [B, C, H, W]

        return imageBatched
            ? warped
            : engine.Reshape(warped, [channels, height, width]);
    }

    /// <summary>
    /// Maps pixel coordinates to the <c>[-1, 1]</c> range GridSample samples in, using the
    /// <c>align_corners=false</c> convention: <c>norm = 2 * (p + 0.5) / extent - 1</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// FIXED: this previously used <c>2p / (extent - 1) - 1</c>, which is the <c>align_corners=TRUE</c>
    /// mapping, while the two-argument <c>IEngine.GridSample(input, grid)</c> it feeds is documented as a
    /// torchvision-default shim — and torchvision defaults to <c>align_corners=false</c>, placing
    /// normalized -1 and +1 at the OUTER EDGES of the border pixels rather than at their centres.
    /// </para>
    /// <para>
    /// The mismatch made the sampler decode every requested coordinate <c>p</c> as
    /// <c>p * extent / (extent - 1) - 0.5</c>: a half-pixel shift plus an outward stretch that grows
    /// toward the edges (1.6% at extent 64). Every warp was therefore slightly wrong, in a way no
    /// shape or finiteness check could see and that a "did the warp reduce the error" test still passes
    /// through, because a mostly-right warp still reduces error.
    /// </para>
    /// <para>
    /// MEASURED, not reasoned about: sampling an affine map <c>f(y,x) = x + 10y</c> at intended (3, 5)
    /// returned 51.033333333333335 rather than 53 — exactly <c>2.7 + 10 * 4.8333</c>, the
    /// false-convention decode of true-convention input. See BezierAlignConventionTests.
    /// </para>
    /// </remarks>
    private static Tensor<T> Normalize<T>(
        IEngine engine, Tensor<T> pixels, int extent, INumericOperations<T> numOps)
    {
        if (extent <= 1)
        {
            // Single row or column: every sample must come from that one line's centre.
            return engine.TensorMultiplyScalar(pixels, numOps.Zero);
        }

        var scaled = engine.TensorMultiplyScalar(pixels, numOps.FromDouble(2.0 / extent));
        return engine.TensorAddScalar(scaled, numOps.FromDouble((1.0 / extent) - 1.0));
    }

    /// <summary>
    /// Returns the constant per-pixel x and y coordinate planes for a resolution, shaped
    /// <c>[1, 1, H, W]</c> so they broadcast across the batch.
    /// </summary>
    /// <remarks>
    /// These are constants, not parameters: they carry no gradient, which is why the flow field alone
    /// determines how the warp responds during training.
    /// </remarks>
    private static (Tensor<T> BaseX, Tensor<T> BaseY) GetBaseGrids<T>(
        int height, int width, INumericOperations<T> numOps)
    {
        var cached = BaseGrids.GetOrAdd((height, width, typeof(T)), _ =>
        {
            var x = new Tensor<T>([1, 1, height, width]);
            var y = new Tensor<T>([1, 1, height, width]);
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int index = h * width + w;
                    x[index] = numOps.FromDouble(w);
                    y[index] = numOps.FromDouble(h);
                }
            }

            return (x, y);
        });

        return ((Tensor<T>, Tensor<T>))cached;
    }
}
