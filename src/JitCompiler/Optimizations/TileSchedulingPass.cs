using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Partitions large Conv2D and MatMul operations into cache-optimal tile sizes.
/// Critical for CPU performance — ensures data fits in L1/L2 cache during computation.
/// </summary>
/// <remarks>
/// <para>
/// Modern CPUs have multi-level caches:
/// - L1: ~32-64KB per core (fastest, ~1ns)
/// - L2: ~256KB-1MB per core (~5ns)
/// - L3: ~6-30MB shared (~15ns)
/// - RAM: GBs (~50-100ns)
///
/// Without tiling, large Conv2D/MatMul operations thrash the cache — data gets evicted
/// before it can be reused. Tiling partitions the work so each tile fits in L2 cache,
/// maximizing data reuse and reducing memory bandwidth requirements.
/// </para>
/// <para>
/// PyTorch TorchInductor does this via Triton tile sizes. We do it at the IR level
/// by annotating operations with optimal tile dimensions based on tensor sizes and
/// detected cache sizes.
/// </para>
/// </remarks>
public class TileSchedulingPass : IOptimizationPass
{
    /// <summary>
    /// Target L2 cache size in bytes. Default: 256KB (conservative for most CPUs).
    /// </summary>
    public int TargetCacheSizeBytes { get; set; } = 256 * 1024;

    /// <summary>
    /// Element size in bytes. Default: 8 (double precision).
    /// </summary>
    public int ElementSizeBytes { get; set; } = 8;

    public string Name => "TileScheduling";

    public IRGraph Optimize(IRGraph graph)
    {
        int elementsPerCacheLine = TargetCacheSizeBytes / ElementSizeBytes;

        foreach (var op in graph.Operations)
        {
            switch (op)
            {
                case Conv2DOp conv:
                    AnnotateConvTiling(conv, graph.TensorShapes, elementsPerCacheLine);
                    break;
                case MatMulOp matmul:
                    AnnotateMatMulTiling(matmul, graph.TensorShapes, elementsPerCacheLine);
                    break;
                case FusedConv2DBiasActivationOp fusedConv:
                    AnnotateFusedConvTiling(fusedConv, graph.TensorShapes, elementsPerCacheLine);
                    break;
            }
        }

        return graph;
    }

    /// <summary>
    /// Annotates a Conv2D operation with optimal tile sizes for cache utilization.
    /// </summary>
    private void AnnotateConvTiling(Conv2DOp conv, Dictionary<int, int[]> shapes, int cacheElements)
    {
        if (!shapes.TryGetValue(conv.InputIds[0], out var inputShape) || inputShape.Length < 4)
            return;

        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        // Tile height: how many rows of the output can we process while keeping
        // input + kernel + output tiles in L2 cache
        // Each output row needs: kernel_h rows of input * width * channels elements
        // Plus the output row itself: width * out_channels elements
        int kernelSize = 3; // most common
        int inputRowElements = width * channels;
        int outputRowElements = width * channels; // approximate

        // Target: (tileH * kernelSize * inputRowElements + tileH * outputRowElements) <= cacheElements
        int elementsPerTileRow = kernelSize * inputRowElements + outputRowElements;
        int tileH = Math.Max(1, cacheElements / elementsPerTileRow);
        tileH = Math.Min(tileH, height); // don't exceed actual height

        // Tile width: for very wide tensors, also tile along width
        int tileW = width; // default: full width
        if (width * channels > cacheElements / 4)
        {
            tileW = Math.Max(1, cacheElements / (4 * channels));
            tileW = Math.Min(tileW, width);
        }

        // Store tiling info in operation metadata
        conv.Metadata["TileH"] = tileH;
        conv.Metadata["TileW"] = tileW;
        conv.Metadata["TileC"] = Math.Min(channels, Math.Max(1, cacheElements / (tileH * tileW)));
    }

    /// <summary>
    /// Annotates a MatMul operation with optimal tile sizes (BLIS-style blocking).
    /// </summary>
    private void AnnotateMatMulTiling(MatMulOp matmul, Dictionary<int, int[]> shapes, int cacheElements)
    {
        if (!shapes.TryGetValue(matmul.InputIds[0], out var aShape) || aShape.Length < 2)
            return;
        if (!shapes.TryGetValue(matmul.InputIds[1], out var bShape) || bShape.Length < 2)
            return;

        int m = aShape[^2]; // rows of A
        int k = aShape[^1]; // cols of A = rows of B
        int n = bShape[^1]; // cols of B

        // BLIS-style blocking: tile M, N, K to fit in L2
        // Each tile processes: mc x kc block of A, kc x nc block of B, mc x nc block of C
        // Memory: mc*kc + kc*nc + mc*nc <= cacheElements
        // Optimal: mc ≈ nc ≈ sqrt(cacheElements/3), kc ≈ cacheElements/(3*mc)
        int tileSide = (int)Math.Sqrt(cacheElements / 3.0);
        int mc = Math.Min(m, Math.Max(1, tileSide));
        int nc = Math.Min(n, Math.Max(1, tileSide));
        int kc = Math.Min(k, Math.Max(1, cacheElements / (3 * Math.Max(1, mc))));

        matmul.Metadata["TileM"] = mc;
        matmul.Metadata["TileN"] = nc;
        matmul.Metadata["TileK"] = kc;
    }

    /// <summary>
    /// Annotates a fused Conv2D+Bias+Activation with tile sizes.
    /// </summary>
    private void AnnotateFusedConvTiling(FusedConv2DBiasActivationOp fusedConv,
        Dictionary<int, int[]> shapes, int cacheElements)
    {
        if (!shapes.TryGetValue(fusedConv.InputIds[0], out var inputShape) || inputShape.Length < 4)
            return;

        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        // Same tiling logic as Conv2D — the fusion doesn't change the data access pattern
        int inputRowElements = width * channels;
        int outputRowElements = width * channels;
        int elementsPerTileRow = 3 * inputRowElements + outputRowElements;
        int tileH = Math.Max(1, Math.Min(height, cacheElements / elementsPerTileRow));

        fusedConv.Metadata["TileH"] = tileH;
        fusedConv.Metadata["TileW"] = width;
    }
}
