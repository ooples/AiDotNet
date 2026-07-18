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

    /// <summary>
    /// Faithful path: the sharded model reproduces a real trained transformer's blocks — RMSNorm with a trained
    /// per-layer γ and a GELU feed-forward activation — so tensor-parallel serving (ws 2 and 4) still produces the
    /// SAME generated tokens as ws1. Proves the faithful norm/activation math is sharding-invariant end-to-end.
    /// </summary>
    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelPagedServing_Faithful_RmsNormGelu_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        var rng = RandomHelper.CreateSeededRandom(13579);
        var embedding = RandomTensor(rng, Vocab, EmbedDim);
        var lmHead = RandomTensor(rng, Vocab, EmbedDim);
        var finalGamma = RandomGamma(rng, EmbedDim);
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
                Norm1Gamma = RandomGamma(rng, EmbedDim), Norm2Gamma = RandomGamma(rng, EmbedDim),
            };
        }

        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        int[] reference = GenerateThroughBatcher(1, embedding, lmHead, layers, prompt, maxNew, finalGamma);
        int[] sharded = GenerateThroughBatcher(worldSize, embedding, lmHead, layers, prompt, maxNew, finalGamma);

        Assert.Equal(maxNew, reference.Length);
        Assert.Equal(reference, sharded);
    }

    /// <summary>
    /// Grouped-query attention (numKVHeads &lt; numHeads): the KV heads are narrower than the query heads and each
    /// query head attends to its KV group. Tensor-parallel serving (world size 2) still produces the SAME tokens as
    /// world size 1 — the KV-head sharding + query-to-group mapping are sharding-invariant.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task TensorParallelPagedServing_GroupedQueryAttention_MatchesUnsharded()
    {
        await Task.Yield();

        const int kvHeads = 2; // NumHeads=4 query heads -> 2 KV heads (2 query heads per group)
        int headDim = EmbedDim / NumHeads;
        int kvDim = kvHeads * headDim;

        var rng = RandomHelper.CreateSeededRandom(24681);
        var embedding = RandomTensor(rng, Vocab, EmbedDim);
        var lmHead = RandomTensor(rng, Vocab, EmbedDim);
        var finalGamma = RandomGamma(rng, EmbedDim);
        var layers = new TensorParallelLayerWeights<double>[NumLayers];
        for (int l = 0; l < NumLayers; l++)
            layers[l] = new TensorParallelLayerWeights<double>
            {
                QWeight = RandomTensor(rng, EmbedDim, EmbedDim), QBias = RandomTensor(rng, EmbedDim),
                KWeight = RandomTensor(rng, kvDim, EmbedDim), KBias = RandomTensor(rng, kvDim),   // narrower KV
                VWeight = RandomTensor(rng, kvDim, EmbedDim), VBias = RandomTensor(rng, kvDim),
                OWeight = RandomTensor(rng, EmbedDim, EmbedDim), OBias = RandomTensor(rng, EmbedDim),
                UpWeight = RandomTensor(rng, FfnDim, EmbedDim), UpBias = RandomTensor(rng, FfnDim),
                DownWeight = RandomTensor(rng, EmbedDim, FfnDim), DownBias = RandomTensor(rng, EmbedDim),
                Norm1Gamma = RandomGamma(rng, EmbedDim), Norm2Gamma = RandomGamma(rng, EmbedDim),
            };

        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        int[] reference = GenerateThroughBatcher(1, embedding, lmHead, layers, prompt, maxNew, finalGamma, numKVHeads: kvHeads);
        int[] sharded = GenerateThroughBatcher(2, embedding, lmHead, layers, prompt, maxNew, finalGamma, numKVHeads: kvHeads);

        Assert.Equal(maxNew, reference.Length);
        Assert.Equal(reference, sharded);
    }

    /// <summary>
    /// Gated SwiGLU FFN (LLaMA/Mistral/Qwen2: <c>Down(act(Gate(x)) * Up(x))</c>): the gate and up projections
    /// are column-partitioned across ranks and multiplied per-slice, so tensor-parallel serving (world sizes 2
    /// and 4) produces the SAME tokens as world size 1. Proves the gate sharding is invariant.
    /// </summary>
    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelPagedServing_GatedSwiGLU_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        var rng = RandomHelper.CreateSeededRandom(13579);
        var embedding = RandomTensor(rng, Vocab, EmbedDim);
        var lmHead = RandomTensor(rng, Vocab, EmbedDim);
        var finalGamma = RandomGamma(rng, EmbedDim);
        var layers = new TensorParallelLayerWeights<double>[NumLayers];
        for (int l = 0; l < NumLayers; l++)
            layers[l] = new TensorParallelLayerWeights<double>
            {
                QWeight = RandomTensor(rng, EmbedDim, EmbedDim), QBias = RandomTensor(rng, EmbedDim),
                KWeight = RandomTensor(rng, EmbedDim, EmbedDim), KBias = RandomTensor(rng, EmbedDim),
                VWeight = RandomTensor(rng, EmbedDim, EmbedDim), VBias = RandomTensor(rng, EmbedDim),
                OWeight = RandomTensor(rng, EmbedDim, EmbedDim), OBias = RandomTensor(rng, EmbedDim),
                UpWeight = RandomTensor(rng, FfnDim, EmbedDim), UpBias = RandomTensor(rng, FfnDim),
                GateWeight = RandomTensor(rng, FfnDim, EmbedDim), GateBias = RandomTensor(rng, FfnDim), // gated
                DownWeight = RandomTensor(rng, EmbedDim, FfnDim), DownBias = RandomTensor(rng, EmbedDim),
                Norm1Gamma = RandomGamma(rng, EmbedDim), Norm2Gamma = RandomGamma(rng, EmbedDim),
            };

        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        int[] reference = GenerateThroughBatcher(1, embedding, lmHead, layers, prompt, maxNew, finalGamma);
        int[] sharded = GenerateThroughBatcher(worldSize, embedding, lmHead, layers, prompt, maxNew, finalGamma);

        Assert.Equal(maxNew, reference.Length);
        Assert.Equal(reference, sharded);
    }

    // GELU (tanh approximation), matching a trained transformer's feed-forward activation. Any fixed activation
    // proves sharding-invariance since ws1 and wsN use the same function; GELU exercises a non-piecewise-linear op.
    private static double Gelu(double v)
        => 0.5 * v * (1.0 + System.Math.Tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));

    private static Tensor<double> RandomGamma(System.Random rng, int dim)
    {
        var t = new Tensor<double>(new[] { dim });
        for (int i = 0; i < dim; i++) t[i] = 0.5 + rng.NextDouble(); // positive scale around 1
        return t;
    }

    private static int[] GenerateThroughBatcher(
        int worldSize, Tensor<double> embedding, Tensor<double> lmHead, TensorParallelLayerWeights<double>[] layers,
        int[] prompt, int maxNew, Tensor<double>? finalNormGamma = null, int? numKVHeads = null)
    {
        var model = finalNormGamma is null
            ? new TensorParallelPagedModel<double>(worldSize, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab, numKVHeads: numKVHeads)
            : new TensorParallelPagedModel<double>(
                worldSize, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab,
                useRmsNorm: true, finalNormGamma: finalNormGamma, ffnActivation: Gelu, numKVHeads: numKVHeads);
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
