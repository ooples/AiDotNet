using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Runs a tensor-parallel forward pass for inference (serving), executing all tensor-parallel ranks so their
/// collective operations (column-parallel gather, row-parallel all-reduce) rendezvous and combine into the
/// same fully-reduced output every rank sees.
/// </summary>
/// <remarks>
/// <para>
/// Tensor parallelism shards each linear layer's weights across ranks (Megatron style): a
/// <see cref="Layers.ColumnParallelLinear{T}"/> owns a slice of the output columns, and a following
/// <see cref="Layers.RowParallelLinear{T}"/> owns the matching input rows and all-reduces the partial sums.
/// This runner drives every rank's forward and returns the combined result — the model output is identical to
/// running the un-sharded model, but the per-rank compute and weight memory are divided across the ranks.
/// </para>
/// <para>
/// <see cref="ForwardInProcess{T}"/> simulates the ranks in ONE process (one task per rank over an
/// <see cref="InMemoryCommunicationBackend{T}"/>), which is how single-node multi-device / test scenarios run
/// and how the sharded-equals-unsharded equivalence is verified. A real multi-node deployment instead runs one
/// process per rank against a networked backend (MPI/NCCL/Gloo) and calls the same layer forwards.
/// </para>
/// <para><b>For Beginners:</b> A very large model may not fit (or run fast enough) on a single GPU. Tensor
/// parallelism splits each layer's weights across several GPUs so they share the work on every token. This
/// helper runs all those splits together and stitches the answer back into exactly what the whole model would
/// have produced.
/// </para>
/// </remarks>
public static class TensorParallelInference
{
    /// <summary>
    /// Runs a tensor-parallel forward across <paramref name="worldSize"/> ranks in-process (one task per rank
    /// over a shared <see cref="InMemoryCommunicationBackend{T}"/>) and returns the combined output. Each rank's
    /// layer stack is built by <paramref name="buildRank"/> from that rank's communication backend; the stack's
    /// collective ops rendezvous across the tasks. All ranks produce the same fully-reduced output, which is
    /// returned.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="worldSize">Number of tensor-parallel ranks (&gt;= 1).</param>
    /// <param name="input">The input tensor, replicated to every rank.</param>
    /// <param name="buildRank">Builds this rank's sharded layer stack from its communication backend (e.g. a
    /// <see cref="Layers.ColumnParallelLinear{T}"/> followed by a <see cref="Layers.RowParallelLinear{T}"/>).</param>
    /// <returns>The tensor-parallel forward output (identical across ranks).</returns>
    public static Tensor<T> ForwardInProcess<T>(
        int worldSize,
        Tensor<T> input,
        Func<ICommunicationBackend<T>, IReadOnlyList<ILayer<T>>> buildRank)
    {
        if (worldSize < 1)
            throw new ArgumentOutOfRangeException(nameof(worldSize), worldSize, "worldSize must be >= 1.");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (buildRank is null) throw new ArgumentNullException(nameof(buildRank));

        string environmentId = Guid.NewGuid().ToString();
        var outputs = new Tensor<T>[worldSize];
        var errors = new Exception?[worldSize];
        var tasks = new Task[worldSize];

        for (int r = 0; r < worldSize; r++)
        {
            int rank = r;
            tasks[rank] = Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<T>(rank, worldSize, environmentId);
                backend.Initialize();
                try
                {
                    var layers = buildRank(backend);
                    var x = input;
                    for (int i = 0; i < layers.Count; i++)
                    {
                        x = layers[i].Forward(x);
                    }
                    outputs[rank] = x;
                }
                catch (Exception ex)
                {
                    // Capture per-rank failures so a collective-op deadlock/exception on one rank surfaces
                    // as a real error instead of hanging the other ranks' barriers indefinitely.
                    errors[rank] = ex;
                }
                finally
                {
                    backend.Shutdown();
                }
            });
        }

        Task.WaitAll(tasks);

        for (int r = 0; r < worldSize; r++)
        {
            if (errors[r] is { } ex)
                throw new InvalidOperationException($"Tensor-parallel rank {r} failed during the forward pass.", ex);
        }

        // Every rank holds the same fully-reduced output (row-parallel all-reduce / column-parallel gather).
        return outputs[0];
    }
}
