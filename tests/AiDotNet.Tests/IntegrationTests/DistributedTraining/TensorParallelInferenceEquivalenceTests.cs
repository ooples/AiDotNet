using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.DistributedTraining;
using AiDotNet.DistributedTraining.Layers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Proves tensor-parallel serving is TRANSPARENT: a Megatron-style MLP block
/// (<see cref="ColumnParallelLinear{T}"/> up-projection + ReLU, then <see cref="RowParallelLinear{T}"/>
/// down-projection with an all-reduce) sharded across 2 and 4 ranks produces the SAME forward output as the
/// un-sharded (world-size 1) model, when every rank is seeded from the SAME full weights. This is the correct
/// way to validate tensor parallelism without N physical GPUs: shard-count invariance == correctness. The
/// ranks run concurrently over an in-process communication backend so their all-reduce actually rendezvous.
/// </summary>
public sealed class TensorParallelInferenceEquivalenceTests
{
    private const int InSize = 8;
    private const int Hidden = 12;   // divisible by 2 and 4 for clean shards (ShardCount also handles remainders)
    private const int OutSize = 8;
    private const int Batch = 2;
    private const double Tol = 1e-9;

    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelMlp_ShardedForward_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        // Shared FULL weights every rank shards from (deterministic).
        var rng = RandomHelper.CreateSeededRandom(1234);
        var upW = RandomTensor(rng, Hidden, InSize);   // ColumnParallel weight layout [outputSize, inputSize]
        var upB = RandomTensor(rng, Hidden);
        var downW = RandomTensor(rng, OutSize, Hidden); // RowParallel weight layout [outputSize, inputSize]
        var downB = RandomTensor(rng, OutSize);

        var input = RandomTensor(rng, Batch, InSize);

        IReadOnlyList<ILayer<double>> BuildRank(ICommunicationBackend<double> backend)
        {
            // Up-projection: column-parallel (each rank owns a slice of the hidden columns), ReLU applied
            // locally, output NOT gathered so it feeds the row-parallel down-projection's matching input shard.
            var up = new ColumnParallelLinear<double>(
                backend, InSize, Hidden, gatherOutput: false, activationFunction: new ReLUActivation<double>());
            up.SetFromFullWeights(upW, upB);

            // Down-projection: row-parallel (each rank owns the matching input rows), all-reduces the partials.
            var down = new RowParallelLinear<double>(backend, Hidden, OutSize);
            down.SetFromFullWeights(downW, downB);

            return new ILayer<double>[] { up, down };
        }

        // Un-sharded reference: world-size 1 (one rank owns everything, no split).
        var reference = TensorParallelInference.ForwardInProcess(1, input, BuildRank);
        // Sharded run across `worldSize` ranks.
        var sharded = TensorParallelInference.ForwardInProcess(worldSize, input, BuildRank);

        Assert.Equal(reference.Shape.Length, sharded.Shape.Length);
        Assert.Equal(Batch, sharded.Shape[0]);
        Assert.Equal(OutSize, sharded.Shape[1]);

        var refSpan = reference.AsSpan();
        var shardSpan = sharded.AsSpan();
        Assert.Equal(refSpan.Length, shardSpan.Length);
        for (int i = 0; i < refSpan.Length; i++)
        {
            Assert.Equal(refSpan[i], shardSpan[i], Tol);
        }
    }

    [Theory(Timeout = 120000)]
    [InlineData(2, false)]
    [InlineData(4, false)]
    [InlineData(2, true)]  // causal
    public async Task TensorParallelAttention_ShardedForward_MatchesUnsharded(int worldSize, bool causal)
    {
        await Task.Yield();

        const int embedDim = 16;
        const int numHeads = 4; // divisible by 2 and 4
        const int seq = 5;
        const int batch = 2;

        var rng = RandomHelper.CreateSeededRandom(4242);
        var qW = RandomTensor(rng, embedDim, embedDim); var qB = RandomTensor(rng, embedDim);
        var kW = RandomTensor(rng, embedDim, embedDim); var kB = RandomTensor(rng, embedDim);
        var vW = RandomTensor(rng, embedDim, embedDim); var vB = RandomTensor(rng, embedDim);
        var oW = RandomTensor(rng, embedDim, embedDim); var oB = RandomTensor(rng, embedDim);

        var input = RandomTensor(rng, batch, seq, embedDim);

        IReadOnlyList<ILayer<double>> BuildRank(ICommunicationBackend<double> backend)
        {
            var attn = new TensorParallelAttention<double>(backend, embedDim, numHeads, causal);
            attn.SetFromFullWeights(qW, qB, kW, kB, vW, vB, oW, oB);
            return new ILayer<double>[] { attn };
        }

        var reference = TensorParallelInference.ForwardInProcess(1, input, BuildRank);
        var sharded = TensorParallelInference.ForwardInProcess(worldSize, input, BuildRank);

        var refSpan = reference.AsSpan();
        var shardSpan = sharded.AsSpan();
        Assert.Equal(refSpan.Length, shardSpan.Length);
        for (int i = 0; i < refSpan.Length; i++)
        {
            Assert.Equal(refSpan[i], shardSpan[i], Tol);
        }
    }

    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelTransformerBlock_ShardedForward_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        const int embedDim = 16;
        const int numHeads = 4;
        const int ffnDim = 32;
        const int seq = 4;
        const int batch = 2;

        var rng = RandomHelper.CreateSeededRandom(9001);
        var qW = RandomTensor(rng, embedDim, embedDim); var qB = RandomTensor(rng, embedDim);
        var kW = RandomTensor(rng, embedDim, embedDim); var kB = RandomTensor(rng, embedDim);
        var vW = RandomTensor(rng, embedDim, embedDim); var vB = RandomTensor(rng, embedDim);
        var oW = RandomTensor(rng, embedDim, embedDim); var oB = RandomTensor(rng, embedDim);
        var upW = RandomTensor(rng, ffnDim, embedDim); var upB = RandomTensor(rng, ffnDim);
        var downW = RandomTensor(rng, embedDim, ffnDim); var downB = RandomTensor(rng, embedDim);

        var input = RandomTensor(rng, batch, seq, embedDim);

        IReadOnlyList<ILayer<double>> BuildRank(ICommunicationBackend<double> backend)
        {
            var block = new TensorParallelTransformerBlock<double>(backend, embedDim, numHeads, ffnDim, causal: true);
            block.SetFromFullWeights(qW, qB, kW, kB, vW, vB, oW, oB, upW, upB, downW, downB);
            return new ILayer<double>[] { block };
        }

        var reference = TensorParallelInference.ForwardInProcess(1, input, BuildRank);
        var sharded = TensorParallelInference.ForwardInProcess(worldSize, input, BuildRank);

        var refSpan = reference.AsSpan();
        var shardSpan = sharded.AsSpan();
        Assert.Equal(refSpan.Length, shardSpan.Length);
        for (int i = 0; i < refSpan.Length; i++)
        {
            Assert.Equal(refSpan[i], shardSpan[i], Tol);
        }
    }

    private static Tensor<double> RandomTensor(System.Random rng, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.NextDouble() * 2.0 - 1.0;
        }
        return t;
    }
}
