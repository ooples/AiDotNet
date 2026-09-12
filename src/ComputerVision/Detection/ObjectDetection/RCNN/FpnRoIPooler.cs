using AiDotNet.Tensors.Engines;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;

/// <summary>
/// Pools each region of interest from the feature-pyramid level that matches its size.
/// </summary>
/// <remarks>
/// <para>
/// Feature Pyramid Networks (Lin et al. 2017, eq. 1) assign a box of width <c>w</c> and height
/// <c>h</c> (in input pixels) to level <c>k = floor(k0 + log2(sqrt(w h) / 224))</c> with
/// <c>k0 = 4</c>, clamped to the available levels: small boxes read the fine, high-resolution maps
/// and large boxes the coarse ones. This is the rule detectron2 and torchvision implement
/// (canonical box size 224 at canonical level 4).
/// </para>
/// <para>
/// The boxes are sampling coordinates, constant to the gradient as in standard RoIAlign; the pooled
/// features stay on the gradient tape, and the per-level results are put back in the caller's box
/// order with an engine gather.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class FpnRoIPooler<T>
{
    private const double CanonicalBoxSize = 224.0;
    private const int CanonicalLevel = 4;

    /// <summary>
    /// Assigns each box to a pyramid level.
    /// </summary>
    /// <param name="boxes">Boxes <c>[N, 4]</c> as (x1, y1, x2, y2) in input pixels.</param>
    /// <param name="strides">The stride of each available level, finest first (for example 4, 8, 16, 32).</param>
    /// <returns>For each box, the index into <paramref name="strides"/> of its level.</returns>
    internal static int[] AssignLevels(Tensor<T> boxes, IReadOnlyList<int> strides)
    {
        ValidateStrides(strides);
        var ops = MathHelper.GetNumericOperations<T>();
        int minLevel = Log2(strides[0]);
        int maxLevel = Log2(strides[strides.Count - 1]);
        var assignment = new int[boxes.Shape[0]];
        for (int i = 0; i < assignment.Length; i++)
        {
            double w = ops.ToDouble(boxes[i, 2]) - ops.ToDouble(boxes[i, 0]);
            double h = ops.ToDouble(boxes[i, 3]) - ops.ToDouble(boxes[i, 1]);
            double size = Math.Sqrt(Math.Max(w, 0) * Math.Max(h, 0));

            // + 1e-8 as in detectron2, so a degenerate box maps to the finest level rather than -inf.
            int level = (int)Math.Floor(CanonicalLevel + Math.Log(size / CanonicalBoxSize + 1e-8, 2));
            assignment[i] = Math.Min(Math.Max(level, minLevel), maxLevel) - minLevel;
        }

        return assignment;
    }

    /// <summary>
    /// Pools every box from its assigned level.
    /// </summary>
    /// <param name="align">The RoIAlign operator (output size and sampling ratio).</param>
    /// <param name="levels">Pyramid feature maps, finest first, one per stride.</param>
    /// <param name="strides">The stride of each level.</param>
    /// <param name="boxes">Boxes <c>[N, 4]</c> in input pixels.</param>
    /// <returns>Pooled features <c>[N, channels, outputSize, outputSize]</c> in the order of <paramref name="boxes"/>.</returns>
    public static Tensor<T> Pool(RoIAlign<T> align, IReadOnlyList<Tensor<T>> levels, IReadOnlyList<int> strides, Tensor<T> boxes)
    {
        if (strides is null) throw new ArgumentNullException(nameof(strides));
        if (levels.Count != strides.Count)
        {
            throw new ArgumentException(
                $"{levels.Count} pyramid levels but {strides.Count} strides; they must correspond one to one.",
                nameof(strides));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var assignment = AssignLevels(boxes, strides);

        var parts = new List<Tensor<T>>();
        var order = new List<int>(assignment.Length);
        for (int level = 0; level < levels.Count; level++)
        {
            var members = new List<int>();
            for (int i = 0; i < assignment.Length; i++)
            {
                if (assignment[i] == level)
                {
                    members.Add(i);
                }
            }

            if (members.Count == 0)
            {
                continue;
            }

            var subset = new Tensor<T>(new[] { members.Count, 4 });
            for (int m = 0; m < members.Count; m++)
            {
                for (int c = 0; c < 4; c++)
                {
                    subset[m, c] = ops.FromDouble(ops.ToDouble(boxes[members[m], c]));
                }
            }

            parts.Add(align.Forward(levels[level], subset, 1.0 / strides[level]));
            order.AddRange(members);
        }

        var pooled = parts.Count == 1 ? parts[0] : AiDotNetEngine.Current.TensorConcatenate(parts.ToArray(), 0);

        // pooled row r holds box order[r]; gather it back so row i holds box i.
        var positionOf = new int[order.Count];
        bool identity = true;
        for (int r = 0; r < order.Count; r++)
        {
            positionOf[order[r]] = r;
            identity &= order[r] == r;
        }

        return identity ? pooled : CvTensorOps<T>.Select(pooled, positionOf, 0);
    }

    private static void ValidateStrides(IReadOnlyList<int> strides)
    {
        if (strides is null) throw new ArgumentNullException(nameof(strides));
        if (strides.Count == 0)
        {
            throw new ArgumentException("At least one pyramid stride is required.", nameof(strides));
        }

        int previous = 0;
        for (int i = 0; i < strides.Count; i++)
        {
            int stride = strides[i];
            if (stride <= 0 || (stride & (stride - 1)) != 0)
            {
                throw new ArgumentException(
                    $"Pyramid strides must be positive powers of two; got {stride}.", nameof(strides));
            }

            if (i > 0 && stride != 2L * previous)
            {
                throw new ArgumentException(
                    "Each pyramid stride must be exactly double the previous stride.", nameof(strides));
            }

            previous = stride;
        }
    }

    private static int Log2(int stride)
    {
        // Shift the value down, rather than shifting 1 past the signed-int boundary.
        // The latter wraps its shift count and can loop forever on a large invalid stride.
        int level = 0;
        while (stride > 1)
        {
            stride >>= 1;
            level++;
        }

        return level;
    }
}
