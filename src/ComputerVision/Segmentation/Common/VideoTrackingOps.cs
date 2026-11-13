using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Mask resampling and per-object statistics shared by the memory-based video segmentation models (DEVA, UniVS),
/// built from whole-tensor engine operations.
/// </summary>
internal static class VideoTrackingOps
{
    /// <summary>Minimum foreground pixels for an object to count as visible.</summary>
    internal const int MinVisibleArea = 4;

    /// <summary>Resamples per-object maps <c>[O, H, W]</c> to <c>[O, height, width]</c>.</summary>
    internal static Tensor<T> Resize<T>(Tensor<T> maps, int height, int width, InterpolateMode mode)
    {
        if (maps.Shape[1] == height && maps.Shape[2] == width) return maps;
        var engine = AiDotNetEngine.Current;
        int o = maps.Shape[0];
        var resized = engine.Interpolate(engine.Reshape(maps, new[] { 1, o, maps.Shape[1], maps.Shape[2] }),
            new[] { height, width }, mode, false);
        return engine.Reshape(resized, new[] { o, height, width });
    }

    /// <summary>
    /// Normalizes object masks to <c>[O, H, W]</c>: a single <c>[H, W]</c> mask becomes one object.
    /// </summary>
    internal static Tensor<T> AsObjectMasks<T>(Tensor<T> masks, string paramName)
    {
        if (masks is null) throw new ArgumentNullException(paramName);
        if (masks.Rank == 2) return AiDotNetEngine.Current.Reshape(masks, new[] { 1, masks.Shape[0], masks.Shape[1] });
        if (masks.Rank != 3) throw new ArgumentException("Object masks must be [O, H, W] or [H, W].", paramName);
        return masks;
    }

    /// <summary>
    /// Per-object confidence (mean probability over the pixels at or above 0.5, 0 when there are none) and
    /// visibility (at least <see cref="MinVisibleArea"/> such pixels) of probabilities <c>[O, H, W]</c>.
    /// </summary>
    internal static (double[] Confidences, bool[] Visible) Statistics<T>(Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int o = probabilities.Shape[0];
        if (o == 0) return (Array.Empty<double>(), Array.Empty<bool>());
        int hw = probabilities.Length / o;
        var flat = engine.Reshape(probabilities, new[] { o, hw });
        var foreground = engine.TensorGreaterEqual(flat, Tensor<T>.CreateDefault(new[] { o, hw }, ops.FromDouble(0.5)));
        // Areas in int (a 0/1 sum in Half stops at 2048); the mean divides before it sums, so it cannot overflow.
        var area = engine.ReduceSum(foreground.Cast<int>(), new[] { 1 }, false);
        var confidence = MaskClassificationOps.MeanOverMask(flat, foreground, area);
        int[] areas = area.ToArray();
        var visible = new bool[o];
        for (int i = 0; i < o; i++) visible[i] = areas[i] >= MinVisibleArea;
        return (confidence.Cast<double>().ToArray(), visible);
    }
}
