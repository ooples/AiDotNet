using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// End-to-end proof of FULL paged tensor-parallel serving: driving greedy generation through the actual
/// continuous-batching engine over a <see cref="TensorParallelPagedModel{T}"/> (world size 2 and 4) with a
/// <see cref="CompositePagedKVCache{T}"/> fanning to the per-rank paged caches produces the SAME generated
/// tokens as the un-sharded (world size 1) model. This wires the whole stack: scheduler -> composite per-rank
/// KV -> tensor-parallel paged decode.
/// </summary>
public sealed class TensorParallelPagedServingEquivalenceTests
{
    private const int EmbedDim = 16;
    private const int NumHeads = 4;
    private const int NumLayers = 2;
    private const int FfnDim = 32;
    private const int Vocab = 20;

    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelPagedServing_GreedyThroughBatcher_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        var rng = RandomHelper.CreateSeededRandom(24680);
        var embedding = RandomTensor(rng, Vocab, EmbedDim);
        var lmHead = RandomTensor(rng, Vocab, EmbedDim);
        var layers = new TensorParallelLayerWeights<double>[NumLayers];
        for (int l = 0; l < NumLayers; l++)
        {
            layers[l] = new TensorParallelLayerWeights<double>
            {
                QWeight = RandomTensor(rng, EmbedDim, EmbedDim), QBias = RandomTensor(rng, EmbedDim),
                KWeight = RandomTensor(rng, EmbedDim, EmbedDim), KBias = RandomTensor(rng, EmbedDim),
                VWeight = RandomTensor(rng, EmbedDim, EmbedDim), VBias = RandomTensor(rng, EmbedDim),
                OWeight = RandomTensor(rng, EmbedDim, EmbedDim), OBias = RandomTensor(rng, EmbedDim),
                UpWeight = RandomTensor(rng, FfnDim, EmbedDim), UpBias = RandomTensor(rng, FfnDim),
                DownWeight = RandomTensor(rng, EmbedDim, FfnDim), DownBias = RandomTensor(rng, EmbedDim),
            };
        }

        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        int[] reference = GenerateThroughBatcher(1, embedding, lmHead, layers, prompt, maxNew);
        int[] sharded = GenerateThroughBatcher(worldSize, embedding, lmHead, layers, prompt, maxNew);

        Assert.Equal(maxNew, reference.Length);
        Assert.Equal(reference, sharded); // greedy tokens identical (logits match to fp; argmax stable)
    }

    private static int[] GenerateThroughBatcher(
        int worldSize, Tensor<double> embedding, Tensor<double> lmHead, TensorParallelLayerWeights<double>[] layers,
        int[] prompt, int maxNew)
    {
        var model = new TensorParallelPagedModel<double>(worldSize, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab);
        model.SetFromFullWeights(embedding, lmHead, layers);
        var composite = new CompositePagedKVCache<double>(model.RankCaches);

        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999, EnableSpeculativeDecoding = false };
        using var batcher = new ContinuousBatcher<double>(config, model, composite);

        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int>(prompt),
            MaxNewTokens = maxNew,
            Temperature = 0f // greedy
        };
        var task = batcher.GenerateAsync(request);
        int guard = maxNew + prompt.Length + 16;
        while (!task.IsCompleted && guard-- > 0) batcher.Step();
        Assert.True(task.IsCompleted, "generation should complete within the step budget");
        return task.GetAwaiter().GetResult().GeneratedTokens.ToArray();
    }

    private static Tensor<double> RandomTensor(System.Random rng, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        if (shape.Length == 1)
        {
            for (int i = 0; i < shape[0]; i++) t[i] = rng.NextDouble() * 2.0 - 1.0;
        }
        else
        {
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < shape[1]; j++) t[i, j] = rng.NextDouble() * 2.0 - 1.0;
        }
        return t;
    }
}
