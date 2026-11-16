namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The serialized orders of one point-cloud level: per curve, the code of every point, the order
/// that sorts the points by code, and its inverse.
/// </summary>
/// <remarks>
/// Mirrors Pointcept's <c>Point.serialization</c>: <c>code = batch &lt;&lt; (3 * depth) | curve(grid)</c>,
/// <c>order = argsort(code)</c>, <c>inverse[order[j]] = j</c>. Codes are <see cref="long"/> and
/// orders <see cref="int"/>, never the network's numeric type, so no code is ever rounded.
/// </remarks>
internal sealed class PointSerializationResult
{
    internal PointSerializationResult(long[][] codes, int[][] orders, int[][] inverses, int depth)
    {
        Codes = codes;
        Orders = orders;
        Inverses = inverses;
        Depth = depth;
    }

    internal long[][] Codes { get; }

    internal int[][] Orders { get; }

    internal int[][] Inverses { get; }

    internal int Depth { get; }

    /// <summary>A copy with the curves permuted, as Pointcept's <c>shuffle_orders</c> does.</summary>
    internal PointSerializationResult Permute(int[] permutation)
    {
        var codes = new long[Codes.Length][];
        var orders = new int[Orders.Length][];
        var inverses = new int[Inverses.Length][];
        for (int i = 0; i < permutation.Length; i++)
        {
            codes[i] = Codes[permutation[i]];
            orders[i] = Orders[permutation[i]];
            inverses[i] = Inverses[permutation[i]];
        }

        return new PointSerializationResult(codes, orders, inverses, Depth);
    }
}

/// <summary>Point-cloud serialization over space-filling curves (PTv3 Sec. 3.1).</summary>
internal static class PointSerialization
{
    /// <summary>
    /// The depth Pointcept derives from the grid: <c>int(grid_coord.max() + 1).bit_length()</c>.
    /// </summary>
    internal static int DepthFor(int[] grid)
    {
        int max = 0;
        for (int i = 0; i < grid.Length; i++)
            if (grid[i] > max) max = grid[i];
        int depth = BitLength(max + 1);
        if (depth > 16)
            throw new ArgumentException(
                $"The point cloud spans {max + 1} grid cells along one axis, more than the 65536 a 16-bit "
                + "serialization depth can address. Use a larger grid size.", nameof(grid));
        return depth;
    }

    internal static int BitLength(int value)
    {
        int bits = 0;
        while (value > 0)
        {
            bits++;
            value >>= 1;
        }

        return bits;
    }

    /// <summary>Serializes points with each curve.</summary>
    /// <param name="grid">Grid coordinates, <c>[x0, y0, z0, x1, ...]</c>, all non-negative.</param>
    /// <param name="batch">The batch index of every point; points of one cloud must be contiguous.</param>
    /// <param name="depth">Bits per axis.</param>
    /// <param name="curves">The serialization orders.</param>
    internal static PointSerializationResult Serialize(int[] grid, int[] batch, int depth, IReadOnlyList<ISpaceFillingCurve> curves)
    {
        if (grid is null) throw new ArgumentNullException(nameof(grid));
        if (batch is null) throw new ArgumentNullException(nameof(batch));
        if (curves is null || curves.Count == 0) throw new ArgumentException("At least one serialization order is required.", nameof(curves));
        int count = batch.Length;
        if (grid.Length != count * 3) throw new ArgumentException("Grid must hold three coordinates per point.", nameof(grid));

        int maxBatch = 0;
        for (int i = 0; i < count; i++)
            if (batch[i] > maxBatch) maxBatch = batch[i];
        if (depth * 3 + BitLength(maxBatch + 1) > 63)
            throw new ArgumentException("Serialization depth and batch size do not fit in a 64-bit code.", nameof(depth));

        var codes = new long[curves.Count][];
        var orders = new int[curves.Count][];
        var inverses = new int[curves.Count][];
        for (int k = 0; k < curves.Count; k++)
        {
            var curve = curves[k];
            var code = new long[count];
            for (int i = 0; i < count; i++)
                code[i] = ((long)batch[i] << (depth * 3)) | curve.Encode(grid[i * 3], grid[i * 3 + 1], grid[i * 3 + 2], depth);

            var order = StableArgsort(code);
            var inverse = new int[count];
            for (int j = 0; j < count; j++) inverse[order[j]] = j;
            codes[k] = code;
            orders[k] = order;
            inverses[k] = inverse;
        }

        return new PointSerializationResult(codes, orders, inverses, depth);
    }

    /// <summary>
    /// Argsort that breaks ties by point index, so two points sharing a voxel always serialize in
    /// the same order. <c>torch.argsort</c> leaves ties unspecified.
    /// </summary>
    internal static int[] StableArgsort(long[] keys)
    {
        var order = new int[keys.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int byKey = keys[a].CompareTo(keys[b]);
            return byKey != 0 ? byKey : a.CompareTo(b);
        });
        return order;
    }
}
