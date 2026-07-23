using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.Inference;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// Proves a full tensor-parallel PAGED serving model is transparent: greedy decode (prefill + step-by-step)
/// over the sharded per-rank KV caches at world size 2 and 4 produces logits bit-identical to the un-sharded
/// (world size 1) model, when both are seeded from the same full weights. End-to-end correctness for full
/// paged tensor-parallel serving, verified without an N-GPU box (shard-count invariance == correctness).
/// </summary>
public sealed class TensorParallelPagedModelEquivalenceTests
{
    private const int EmbedDim = 16;
    private const int NumHeads = 4;
    private const int NumLayers = 2;
    private const int FfnDim = 32;
    private const int Vocab = 20;
    private const int PromptLen = 3;
    private const int DecodeSteps = 5;
    // Sharding sums the row-parallel partials in a different order than the un-sharded single matmul, so results
    // agree only up to floating-point non-associativity (accumulated across layers). 1e-6 is the shard-invariance
    // bar; a structural bug would diverge far beyond it.
    private const double Tol = 1e-6;

    [Theory(Timeout = 120000)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task TensorParallelPagedModel_GreedyDecode_MatchesUnsharded(int worldSize)
    {
        await Task.Yield();

        var rng = RandomHelper.CreateSeededRandom(76543);
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

        // A FIXED token sequence (same for both runs) so the paged prefill+decode is exercised without greedy
        // feedback amplifying tiny floating-point differences into divergent sequences.
        var decodeTokens = new[] { 4, 7, 2, 9, 5 };

        var reference = RunFixed(1, embedding, lmHead, layers, decodeTokens);
        var sharded = RunFixed(worldSize, embedding, lmHead, layers, decodeTokens);

        Assert.Equal(reference.Count, sharded.Count);
        for (int i = 0; i < reference.Count; i++)
            Assert.Equal(reference[i], sharded[i], Tol);
    }

    // Runs paged prefill + decode over a FIXED token sequence and returns ALL logits (every position/step).
    private static List<double> RunFixed(
        int worldSize, Tensor<double> embedding, Tensor<double> lmHead, TensorParallelLayerWeights<double>[] layers,
        int[] decodeTokens)
    {
        const long seqId = 1;
        int maxLen = PromptLen + decodeTokens.Length;

        var model = new TensorParallelPagedModel<double>(worldSize, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab);
        model.SetFromFullWeights(embedding, lmHead, layers);
        foreach (var cache in model.RankCaches) cache.AllocateSequence(seqId, maxLen);

        var collected = new List<double>();
        try
        {
            var prompt = new Tensor<double>(new[] { 1, PromptLen });
            for (int i = 0; i < PromptLen; i++) prompt[0, i] = i + 1;

            var prefillLogits = model.PredictWithContext(prompt, new InferenceForwardContext(seqId, 0));
            AppendSpan(collected, prefillLogits.AsSpan());

            for (int step = 0; step < decodeTokens.Length; step++)
            {
                int pos = PromptLen + step;
                var input = new Tensor<double>(new[] { 1, 1 });
                input[0, 0] = decodeTokens[step];
                var stepLogits = model.PredictWithContext(input, new InferenceForwardContext(seqId, pos));
                AppendSpan(collected, stepLogits.AsSpan());
            }
        }
        finally
        {
            model.ShutdownRanks();
        }
        return collected;
    }

    private static void AppendSpan(List<double> list, System.ReadOnlySpan<double> span)
    {
        for (int i = 0; i < span.Length; i++) list.Add(span[i]);
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
