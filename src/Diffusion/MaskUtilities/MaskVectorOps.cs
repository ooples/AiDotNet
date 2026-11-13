namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Whole-vector helpers shared by the mask utilities.
/// </summary>
internal static class MaskVectorOps
{
    /// <summary>
    /// Returns the first <paramref name="length"/> elements of <paramref name="values"/>, padded with
    /// <paramref name="fill"/> when the vector is shorter.
    /// </summary>
    internal static Vector<T> Fit<T>(Vector<T> values, int length, T fill)
    {
        if (values.Length == length)
            return values;
        if (values.Length > length)
            return values.GetSubVector(0, length);
        return Vector<T>.Concatenate(values, DiffusionVectorOps.Full(length - values.Length, fill));
    }

    /// <summary>
    /// Correlates an [height, width] (or [length]) mask with a square odd-sized kernel, replicating the border
    /// pixels (the clamped-index neighbourhood the mask filters use), and returns a tensor of the mask's shape.
    /// </summary>
    internal static Tensor<T> ReplicateFilter2D<T>(Tensor<T> mask, Tensor<T> kernel)
    {
        var engine = AiDotNetEngine.Current;
        int[] shape = mask._shape;
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        int size = kernel._shape[0];
        int half = size / 2;

        var image = engine.Reshape(mask, new[] { 1, 1, height, width });
        var padded = engine.PadNd(image, new[] { half, half, half, half }, PadMode.Replicate, MathHelper.GetNumericOperations<T>().Zero);
        var filtered = engine.Conv2D(padded, engine.Reshape(kernel, new[] { 1, 1, size, size }), 1, 0, 1);
        return engine.Reshape(filtered, shape);
    }

    /// <summary>
    /// Maximum over each size x size neighbourhood of an [height, width] mask, replicating the border pixels.
    /// </summary>
    internal static Tensor<T> ReplicateMaxFilter2D<T>(Tensor<T> mask, int size)
    {
        var engine = AiDotNetEngine.Current;
        int[] shape = mask._shape;
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        int half = size / 2;

        var image = engine.Reshape(mask, new[] { 1, 1, height, width });
        var padded = engine.PadNd(image, new[] { half, half, half, half }, PadMode.Replicate, MathHelper.GetNumericOperations<T>().Zero);
        return engine.Reshape(engine.MaxPool2D(padded, size, 1, 0), shape);
    }

    /// <summary>
    /// A [height, width] tensor that is one inside rows [top, bottom) and columns [left, right) (clipped to the
    /// canvas) and zero elsewhere.
    /// </summary>
    internal static Tensor<T> Rectangle<T>(int height, int width, int top, int left, int bottom, int right)
    {
        return AiDotNetEngine.Current.TensorOuter(Indicator<T>(height, top, bottom), Indicator<T>(width, left, right));
    }

    /// <summary>
    /// A [height, width] tensor that is one where (y - centerY)^2 + (x - centerX)^2 &lt;= radiusSquared.
    /// </summary>
    internal static Tensor<T> Disk<T>(int height, int width, int centerY, int centerX, int radiusSquared)
    {
        var engine = AiDotNetEngine.Current;
        var shape = new[] { height, width };
        // Coordinates are geometry, independent of the mask storage type. Half squares overflow above 255.
        var dy = engine.TensorSquare(engine.TensorArange<double>(-(double)centerY, (double)height - centerY, 1));
        var dx = engine.TensorSquare(engine.TensorArange<double>(-(double)centerX, (double)width - centerX, 1));
        var distanceSq = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.Reshape(dy, new[] { height, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(dx, new[] { 1, width }), shape));
        var limit = Tensor<double>.CreateDefault(shape, radiusSquared);
        return engine.TensorLessEqual(distanceSq, limit).Cast<T>();
    }

    /// <summary>
    /// Sets the rectangle [top, bottom) x [left, right) of <paramref name="mask"/> to one, touching only the
    /// rectangle clipped to the mask (nothing when the clipped rectangle is empty).
    /// </summary>
    internal static Tensor<T> UnionRectangle<T>(Tensor<T> mask, int top, int left, int bottom, int right)
    {
        int height = mask.Shape[0], width = mask.Shape[1];
        int r0 = Math.Max(0, top), r1 = Math.Min(height, bottom);
        int c0 = Math.Max(0, left), c1 = Math.Min(width, right);
        if (r1 <= r0 || c1 <= c0)
            return mask;

        var ones = Tensor<T>.CreateDefault(new[] { r1 - r0, c1 - c0 }, MathHelper.GetNumericOperations<T>().One);
        return AiDotNetEngine.Current.TensorSetSlice(mask, ones, new[] { r0, c0 });
    }

    /// <summary>
    /// Sets the disk (y - centerY)^2 + (x - centerX)^2 &lt;= radius^2 of <paramref name="mask"/> to one, computing
    /// distances only over the disk's bounding square clipped to the mask (nothing when it lies outside).
    /// </summary>
    internal static Tensor<T> UnionDisk<T>(Tensor<T> mask, int centerY, int centerX, int radius)
    {
        int height = mask.Shape[0], width = mask.Shape[1];
        long r0 = Math.Max(0L, (long)centerY - radius), r1 = Math.Min(height, (long)centerY + radius + 1);
        long c0 = Math.Max(0L, (long)centerX - radius), c1 = Math.Min(width, (long)centerX + radius + 1);
        if (r1 <= r0 || c1 <= c0)
            return mask;

        var engine = AiDotNetEngine.Current;
        int rows = (int)(r1 - r0), cols = (int)(c1 - c0);
        var start = new[] { (int)r0, (int)c0 };
        var window = engine.TensorSlice(mask, start, new[] { rows, cols });
        var disk = Disk<T>(rows, cols, (int)(centerY - r0), (int)(centerX - c0), radius * radius);
        return engine.TensorSetSlice(mask, engine.TensorMax(window, disk), start);
    }

    private static Tensor<T> Indicator<T>(int length, int start, int end)
    {
        int from = Math.Max(0, Math.Min(start, length));
        int to = Math.Max(from, Math.Min(end, length));
        var ones = DiffusionVectorOps.Ones<T>(to - from);
        return Vector<T>.Concatenate(new Vector<T>(from), ones, new Vector<T>(length - to)).AsTensor();
    }

    /// <summary>
    /// Linear blend (1 - m) * a + m * b over whole vectors.
    /// </summary>
    internal static Vector<T> Lerp<T>(Vector<T> a, Vector<T> b, Vector<T> weight)
    {
        var engine = AiDotNetEngine.Current;
        var oneMinusWeight = engine.Subtract(DiffusionVectorOps.Ones<T>(weight.Length), weight);
        return engine.Add(engine.Multiply(oneMinusWeight, a), engine.Multiply(weight, b));
    }
}
