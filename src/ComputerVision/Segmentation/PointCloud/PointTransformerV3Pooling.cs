namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// PTv3 grid pooling (Pointcept <c>SerializedPooling</c> / <c>GridPooling</c>): merges the points that
/// fall into one cell of a <c>stride</c>-times coarser grid, max-pooling their projected features.
/// </summary>
/// <remarks>
/// <para>
/// <c>feat = segment_max(Linear(feat)[indices])</c>, <c>coord = segment_mean(coord)</c>, then norm and
/// GELU. m1 clusters by <c>serialized_code &gt;&gt; 3 * log2(stride)</c>, m2 by <c>grid_coord // stride</c>;
/// both are the same partition (the parent cell of a Z-order or Hilbert code is the code shifted
/// right by three bits per level), so both are computed from the coarse grid here. The pooled level
/// is then serialized again: m1 keeps the shifted depth, m2 recomputes it from the coarse grid.
/// </para>
/// <para>The cluster map is kept on the pooled level so unpooling and Sonata's up-casting can scatter
/// features back to the parent points.</para>
/// </remarks>
internal sealed class PointTransformerV3Pooling<T>
{
    private readonly PtContext<T> _ctx;
    private readonly PtLinear<T> _projection;
    private readonly PtNorm<T> _norm;
    private readonly int _stride;
    private readonly PointTransformerV3Revision _revision;

    internal PointTransformerV3Pooling(PtContext<T> ctx, int inputChannels, int outputChannels, int stride,
        PointTransformerV3Revision revision, string name)
    {
        _ctx = ctx;
        _stride = stride;
        _revision = revision;
        _projection = new PtLinear<T>(ctx, inputChannels, outputChannels);
        _norm = revision == PointTransformerV3Revision.M1Base
            ? new PtBatchNorm<T>(ctx, outputChannels, name + ".norm")
            : new PtLayerNorm<T>(ctx, outputChannels);
    }

    internal PointCloudLevel<T> Forward(PointCloudLevel<T> level, IReadOnlyList<ISpaceFillingCurve> curves,
        bool shuffleOrders, bool training)
    {
        int stride = _stride;
        int shift = PointSerialization.BitLength(stride) - 1;
        int depth = level.Serialization.Depth;
        // Pointcept m1: "if pooling_depth > point.serialized_depth: pooling_depth = 0".
        if (_revision == PointTransformerV3Revision.M1Base && shift > depth)
        {
            stride = 1;
            shift = 0;
        }

        int count = level.Count;
        var coarse = new int[count * 3];
        var keys = new long[count];
        for (int i = 0; i < count; i++)
        {
            int x = level.Grid[i * 3] / stride, y = level.Grid[i * 3 + 1] / stride, z = level.Grid[i * 3 + 2] / stride;
            coarse[i * 3] = x;
            coarse[i * 3 + 1] = y;
            coarse[i * 3 + 2] = z;
            keys[i] = ((long)level.Batch[i] << 48) | ((long)x << 32) | ((long)y << 16) | (long)z;
        }

        var sorted = PointSerialization.StableArgsort(keys);
        var cluster = new int[count];
        var heads = new List<int>();
        for (int j = 0; j < count; j++)
        {
            int point = sorted[j];
            if (j == 0 || keys[point] != keys[sorted[j - 1]]) heads.Add(point);
            cluster[point] = heads.Count - 1;
        }

        int pooled = heads.Count;
        var grid = new int[pooled * 3];
        var batch = new int[pooled];
        for (int c = 0; c < pooled; c++)
        {
            int head = heads[c];
            batch[c] = level.Batch[head];
            grid[c * 3] = coarse[head * 3];
            grid[c * 3 + 1] = coarse[head * 3 + 1];
            grid[c * 3 + 2] = coarse[head * 3 + 2];
        }

        var engine = _ctx.Engine;
        var projected = _projection.Forward(level.Features);
        var features = engine.ScatterMax(projected, PtOps.Indices(cluster), out _, 0, pooled);
        features = engine.TensorGELU(_norm.Forward(features, training));

        int newDepth = _revision == PointTransformerV3Revision.M1Base
            ? depth - shift
            : PointSerialization.DepthFor(grid);
        var serialization = PointSerialization.Serialize(grid, batch, newDepth, curves);
        if (shuffleOrders && training) serialization = serialization.Permute(Shuffle(curves.Count));

        return new PointCloudLevel<T>(features, SegmentMean(level.Coordinates, cluster, pooled),
            SegmentMean(level.OriginCoordinates, cluster, pooled), grid, batch, level.BatchCount, serialization)
        {
            Parent = level,
            PoolingInverse = cluster
        };
    }

    internal int[] Shuffle(int count) => PointTransformerV3Encoder<T>.RandomPermutation(_ctx.Random, count);

    private static double[] SegmentMean(double[] values, int[] cluster, int clusters)
    {
        var sums = new double[clusters * 3];
        var counts = new int[clusters];
        for (int i = 0; i < cluster.Length; i++)
        {
            int c = cluster[i];
            counts[c]++;
            sums[c * 3] += values[i * 3];
            sums[c * 3 + 1] += values[i * 3 + 1];
            sums[c * 3 + 2] += values[i * 3 + 2];
        }

        for (int c = 0; c < clusters; c++)
        {
            sums[c * 3] /= counts[c];
            sums[c * 3 + 1] /= counts[c];
            sums[c * 3 + 2] /= counts[c];
        }

        return sums;
    }
}

/// <summary>
/// PTv3 unpooling (Pointcept <c>SerializedUnpooling</c>): projects the coarse features and the skip
/// features to the decoder width and adds the coarse ones to every point of their cell.
/// </summary>
internal sealed class PointTransformerV3Unpooling<T>
{
    private readonly PtContext<T> _ctx;
    private readonly PtLinear<T> _projection;
    private readonly PtNorm<T> _norm;
    private readonly PtLinear<T> _skipProjection;
    private readonly PtNorm<T> _skipNorm;

    internal PointTransformerV3Unpooling(PtContext<T> ctx, int inputChannels, int skipChannels, int outputChannels,
        PointTransformerV3Revision revision, string name)
    {
        _ctx = ctx;
        _projection = new PtLinear<T>(ctx, inputChannels, outputChannels);
        _norm = MakeNorm(ctx, outputChannels, revision, name + ".proj");
        _skipProjection = new PtLinear<T>(ctx, skipChannels, outputChannels);
        _skipNorm = MakeNorm(ctx, outputChannels, revision, name + ".proj_skip");
    }

    internal Tensor<T> Forward(PointCloudLevel<T> coarse, PointCloudLevel<T> parent, bool training)
    {
        var inverse = coarse.PoolingInverse
            ?? throw new InvalidOperationException("Unpooling needs the cluster map its pooling recorded.");
        var engine = _ctx.Engine;
        var up = engine.TensorGELU(_norm.Forward(_projection.Forward(coarse.Features), training));
        var skip = engine.TensorGELU(_skipNorm.Forward(_skipProjection.Forward(parent.Features), training));
        return engine.TensorAdd(skip, engine.TensorIndexSelect(up, PtOps.Indices(inverse), 0));
    }

    private static PtNorm<T> MakeNorm(PtContext<T> ctx, int channels, PointTransformerV3Revision revision, string name)
        => revision == PointTransformerV3Revision.M1Base
            ? new PtBatchNorm<T>(ctx, channels, name + ".norm")
            : new PtLayerNorm<T>(ctx, channels);
}
