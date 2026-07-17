using System.Collections.Generic;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.DistributedTraining.Layers;

/// <summary>
/// Result of partitioning a model's layers for tensor parallelism: the substituted layer list plus a
/// human-readable log of exactly which layers were compute-partitioned and which were left replicated.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class TensorParallelPartitionResult<T>
{
    public required List<ILayer<T>> Layers { get; init; }
    public required int PartitionedLinearCount { get; init; }
    public required int ReplicatedLayerCount { get; init; }
    public required string Log { get; init; }
}

/// <summary>
/// Automatically rewrites a layered model's linear layers into Megatron-LM tensor-parallel primitives
/// (Shoeybi et al. 2019). Unlike a manual Megatron model build, this discovers the substitution from a
/// generic <see cref="ILayer{T}"/> list. It is deliberately CONSERVATIVE ("confirm-safe"): it only
/// substitutes layer shapes whose partitioning is numerically transparent (proven by the
/// AutoPartitioned == NonParallel invariant), and it LEAVES every other layer replicated while logging
/// exactly what happened — so it never silently changes the model's math.
/// </summary>
/// <remarks>
/// <para><b>Substitution rules (all numerically transparent):</b></para>
/// <list type="bullet">
/// <item>A CONSECUTIVE pair of scalar-activated <see cref="FullyConnectedLayer{T}"/> (the classic MLP
/// block) becomes <see cref="ColumnParallelLinear{T}"/> (gatherOutput=false, carrying the first layer's
/// activation) → <see cref="RowParallelLinear{T}"/> (carrying the second layer's activation). This is the
/// Megatron MLP: the intermediate hidden dimension is split across ranks with NO all-gather between the
/// two matmuls; the row-parallel ḡ operator reduces the partial sums exactly once. Because the first
/// layer's activation is element-wise, applying it on each rank's output-column slice equals applying it
/// on the full output and then splitting — so the block is transparent.</item>
/// <item>A LONE scalar-activated <see cref="FullyConnectedLayer{T}"/> becomes a single
/// <see cref="ColumnParallelLinear{T}"/> with gatherOutput=true: each rank computes its slice of the
/// output columns and the tape-aware gather reconstructs the full output — again transparent.</item>
/// <item>Everything else (vector-activated linears, convolutions, norms, attention, activation-only
/// layers, …) is passed through UNCHANGED and counted as replicated.</item>
/// </list>
/// <para>Attention-block partitioning (QKV column-parallel / output row-parallel) is intentionally NOT
/// auto-detected here: recognizing attention structure from a generic layer list is fragile and could
/// silently corrupt gradients. It is a documented future extension, to be added only behind its own
/// transparency invariant.</para>
/// </remarks>
internal static class TensorParallelLayerPartitioner<T>
{
    /// <summary>
    /// Rewrites <paramref name="original"/> into tensor-parallel layers for the given backend/world size.
    /// </summary>
    public static TensorParallelPartitionResult<T> Partition(
        IReadOnlyList<ILayer<T>> original,
        ICommunicationBackend<T> backend)
    {
        if (original is null) throw new System.ArgumentNullException(nameof(original));
        if (backend is null) throw new System.ArgumentNullException(nameof(backend));

        var result = new List<ILayer<T>>(original.Count);
        var log = new StringBuilder();
        int partitioned = 0, replicated = 0;

        int i = 0;
        while (i < original.Count)
        {
            var layer = original[i];

            // A consecutive pair of scalar-activated fully-connected layers -> Megatron MLP (column->row).
            if (i + 1 < original.Count
                && TryGetLinear(layer, out int in0, out int hidden0, out var act0, out var w0, out var b0)
                && TryGetLinear(original[i + 1], out int in1, out int out1, out var act1, out var w1, out var b1)
                && hidden0 == in1)
            {
                var column = new ColumnParallelLinear<T>(backend, in0, hidden0, gatherOutput: false, activationFunction: act0);
                column.SetFromFullWeights(w0, b0);
                var row = new RowParallelLinear<T>(backend, hidden0, out1, activationFunction: act1);
                row.SetFromFullWeights(w1, b1);

                result.Add(column);
                result.Add(row);
                partitioned += 2;
                log.Append($"[{i}] FullyConnected {in0}x{hidden0} -> ColumnParallel; ")
                   .Append($"[{i + 1}] FullyConnected {in1}x{out1} -> RowParallel (Megatron MLP pair)\n");
                i += 2;
                continue;
            }

            // A lone scalar-activated fully-connected layer -> column-parallel with tape-aware gather.
            if (TryGetLinear(layer, out int inL, out int outL, out var actL, out var wL, out var bL))
            {
                var column = new ColumnParallelLinear<T>(backend, inL, outL, gatherOutput: true, activationFunction: actL);
                column.SetFromFullWeights(wL, bL);
                result.Add(column);
                partitioned += 1;
                log.Append($"[{i}] FullyConnected {inL}x{outL} -> ColumnParallel(gather) (lone linear)\n");
                i += 1;
                continue;
            }

            // Not a safely-partitionable linear: keep it replicated, unchanged.
            result.Add(layer);
            replicated += 1;
            log.Append($"[{i}] {layer.GetType().Name} -> replicated (not a scalar-activated FullyConnected)\n");
            i += 1;
        }

        return new TensorParallelPartitionResult<T>
        {
            Layers = result,
            PartitionedLinearCount = partitioned,
            ReplicatedLayerCount = replicated,
            Log = log.ToString(),
        };
    }

    /// <summary>
    /// Recognizes a scalar-activated <see cref="FullyConnectedLayer{T}"/> with a fully materialized weight,
    /// and extracts its (inputSize, outputSize, activation, weight[out,in], bias[out]). Vector-activated or
    /// lazily-unshaped linears return false so the caller leaves them replicated.
    /// </summary>
    private static bool TryGetLinear(
        ILayer<T> layer,
        out int inputSize, out int outputSize,
        out IActivationFunction<T>? activation,
        out Tensor<T> weight, out Tensor<T> bias)
    {
        inputSize = 0; outputSize = 0; activation = null;
        weight = Tensor<T>.Empty(); bias = Tensor<T>.Empty();

        if (layer is not FullyConnectedLayer<T> fc)
            return false;
        // Only scalar (element-wise) activations are safe to apply on a column-split output.
        if (fc.VectorActivation is not null)
            return false;

        var inShape = fc.GetInputShape();
        var outShape = fc.GetOutputShape();
        if (inShape is null || outShape is null || inShape.Length == 0 || outShape.Length == 0)
            return false;
        inputSize = inShape[inShape.Length - 1];
        outputSize = outShape[outShape.Length - 1];
        if (inputSize <= 0 || outputSize <= 0)
            return false;

        // FullyConnectedLayer registers weights [out, in] then biases [out]; GetParameters concatenates in
        // that order. Reconstruct the dense weight/bias tensors to seed the parallel shard.
        var flat = fc.GetParameters();
        long expected = (long)outputSize * inputSize + outputSize;
        if (flat is null || flat.Length != expected)
            return false;

        var w = new Tensor<T>(new[] { outputSize, inputSize });
        int idx = 0;
        for (int o = 0; o < outputSize; o++)
            for (int k = 0; k < inputSize; k++)
                w[o, k] = flat[idx++];
        var bvec = new Tensor<T>(new[] { outputSize });
        for (int o = 0; o < outputSize; o++)
            bvec[o] = flat[idx++];

        activation = fc.ScalarActivation;
        weight = w;
        bias = bvec;
        return true;
    }
}
