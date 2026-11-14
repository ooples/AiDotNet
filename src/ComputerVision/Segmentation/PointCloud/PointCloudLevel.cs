namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// One resolution of a batched point cloud as it flows through Point Transformer V3: the features
/// plus the geometry the attention, sparse convolution and pooling need. Pointcept calls this a
/// <c>Point</c>.
/// </summary>
/// <remarks>
/// Everything geometric (grid coordinates, batch indices, serialization codes, orders, neighbour
/// tables, cluster maps) is integer or double data and never passes through the network's numeric
/// type; only <see cref="Features"/> is a <see cref="Tensor{T}"/>.
/// </remarks>
internal sealed class PointCloudLevel<T>
{
    private readonly Dictionary<int, int[]> _neighbourTables = new();
    private readonly Dictionary<long, SerializedPatchPlan> _patchPlans = new();

    internal PointCloudLevel(Tensor<T> features, double[] coordinates, double[] originCoordinates, int[] grid,
        int[] batch, int batchCount, PointSerializationResult serialization)
    {
        Features = features;
        Coordinates = coordinates;
        OriginCoordinates = originCoordinates;
        Grid = grid;
        Batch = batch;
        BatchCount = batchCount;
        Serialization = serialization;
    }

    /// <summary>Point features, <c>[Count, channels]</c>.</summary>
    internal Tensor<T> Features { get; set; }

    /// <summary>Point coordinates (mean of the merged points after pooling), <c>Count * 3</c>.</summary>
    internal double[] Coordinates { get; }

    /// <summary>Coordinates before augmentation, carried for Sonata's cross-view matching.</summary>
    internal double[] OriginCoordinates { get; }

    /// <summary>Integer grid coordinates at this level's resolution, <c>Count * 3</c>.</summary>
    internal int[] Grid { get; }

    /// <summary>Which cloud of the batch each point belongs to; clouds are contiguous.</summary>
    internal int[] Batch { get; }

    internal int BatchCount { get; }

    internal int Count => Batch.Length;

    internal PointSerializationResult Serialization { get; }

    /// <summary>The finer level this one was pooled from, or null at full resolution.</summary>
    internal PointCloudLevel<T>? Parent { get; set; }

    /// <summary>For every point of <see cref="Parent"/>, the index of the point of this level it merged into.</summary>
    internal int[]? PoolingInverse { get; set; }

    /// <summary>Sonata's masked-point indicator, <c>[Count, 1]</c> of 0/1, or null when nothing is masked.</summary>
    internal Tensor<T>? Mask { get; set; }

    /// <summary>
    /// The submanifold neighbour table for a cubic kernel: entry <c>[i * k^3 + o]</c> is the point at
    /// offset <c>o</c> from point <c>i</c>, or <see cref="Count"/> (a zero row) when that voxel is empty.
    /// </summary>
    /// <remarks>
    /// Equivalent to spconv's <c>SubMConv3d</c> rulebook: outputs exist only at occupied voxels and
    /// each gathers the occupied voxels inside its kernel. Pointcept shares one rulebook per stage
    /// (<c>indice_key</c>); the cache here does the same. When several points share a voxel the
    /// lowest-index one represents it, as the input is expected to be grid-sampled already.
    /// </remarks>
    internal int[] NeighbourTable(int kernelSize)
    {
        if (_neighbourTables.TryGetValue(kernelSize, out var cached)) return cached;
        int radius = kernelSize / 2;
        int volume = kernelSize * kernelSize * kernelSize;
        var occupied = new Dictionary<long, int>(Count);
        for (int i = Count - 1; i >= 0; i--)
            occupied[VoxelKey(Batch[i], Grid[i * 3], Grid[i * 3 + 1], Grid[i * 3 + 2])] = i;

        var table = new int[Count * volume];
        for (int i = 0; i < Count; i++)
        {
            int o = 0;
            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dz = -radius; dz <= radius; dz++, o++)
                    {
                        int x = Grid[i * 3] + dx, y = Grid[i * 3 + 1] + dy, z = Grid[i * 3 + 2] + dz;
                        table[i * volume + o] = x >= 0 && y >= 0 && z >= 0
                            && x <= 0xFFFF && y <= 0xFFFF && z <= 0xFFFF
                            && occupied.TryGetValue(VoxelKey(Batch[i], x, y, z), out int neighbour)
                            ? neighbour
                            : Count;
                    }
                }
            }
        }

        _neighbourTables[kernelSize] = table;
        return table;
    }

    /// <summary>The patch grouping of serialized order <paramref name="orderIndex"/> for a patch size.</summary>
    internal SerializedPatchPlan PatchPlan(int orderIndex, int patchSize)
    {
        long key = ((long)orderIndex << 32) | (uint)patchSize;
        if (_patchPlans.TryGetValue(key, out var cached)) return cached;
        var plan = SerializedPatchPlan.Build(
            Serialization.Orders[orderIndex % Serialization.Orders.Length], Batch, BatchCount, patchSize);
        _patchPlans[key] = plan;
        return plan;
    }

    private static long VoxelKey(int batch, int x, int y, int z)
        => ((long)batch << 48) | ((long)(x & 0xFFFF) << 32) | ((long)(y & 0xFFFF) << 16) | (long)(z & 0xFFFF);
}
