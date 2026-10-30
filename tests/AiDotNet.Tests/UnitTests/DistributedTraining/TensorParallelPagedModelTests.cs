using System;
using System.Collections.Generic;
using AiDotNet.DistributedTraining;
using AiDotNet.Inference;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.DistributedTraining;

/// <summary>
/// Megatron-style tensor parallelism over a paged KV cache (Shoeybi et al. 2019; Kwon et al. 2023): splitting
/// heads and FFN columns across ranks must not change the logits. The serving test project holds the wider
/// equivalence matrix; this is the model's own coverage in the main suite.
/// </summary>
public class TensorParallelPagedModelTests
{
    private const int EmbedDim = 8;
    private const int NumHeads = 4;
    private const int NumLayers = 2;
    private const int FfnDim = 16;
    private const int Vocab = 12;

    private static Tensor<double> Random(Random rng, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++)
            t.SetFlat(i, rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static (Tensor<double> Embedding, Tensor<double> LmHead, TensorParallelLayerWeights<double>[] Layers) Weights(int kvDim)
    {
        var rng = RandomHelper.CreateSeededRandom(2024);
        var layers = new TensorParallelLayerWeights<double>[NumLayers];
        for (int l = 0; l < NumLayers; l++)
        {
            layers[l] = new TensorParallelLayerWeights<double>
            {
                QWeight = Random(rng, EmbedDim, EmbedDim), QBias = Random(rng, EmbedDim),
                KWeight = Random(rng, kvDim, EmbedDim), KBias = Random(rng, kvDim),
                VWeight = Random(rng, kvDim, EmbedDim), VBias = Random(rng, kvDim),
                OWeight = Random(rng, EmbedDim, EmbedDim), OBias = Random(rng, EmbedDim),
                UpWeight = Random(rng, FfnDim, EmbedDim), UpBias = Random(rng, FfnDim),
                DownWeight = Random(rng, EmbedDim, FfnDim), DownBias = Random(rng, EmbedDim),
            };
        }

        return (Random(rng, Vocab, EmbedDim), Random(rng, Vocab, EmbedDim), layers);
    }

    // Prefill three tokens, then decode three fixed tokens; every logit of every position is returned.
    private static List<double> Logits(int worldSize, int? numKVHeads)
    {
        int kvDim = (numKVHeads ?? NumHeads) * (EmbedDim / NumHeads);
        var (embedding, lmHead, layers) = Weights(kvDim);
        var model = new TensorParallelPagedModel<double>(worldSize, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab,
            numKVHeads: numKVHeads);
        model.SetFromFullWeights(embedding, lmHead, layers);
        const long sequence = 1;
        foreach (var cache in model.RankCaches)
            cache.AllocateSequence(sequence, 6);

        var logits = new List<double>();
        try
        {
            var prompt = new Tensor<double>(new[] { 1, 3 });
            for (int i = 0; i < 3; i++)
                prompt[0, i] = i + 1;
            logits.AddRange(model.PredictWithContext(prompt, new InferenceForwardContext(sequence, 0)).ToArray());
            int[] next = { 5, 9, 2 };
            for (int step = 0; step < next.Length; step++)
            {
                var token = new Tensor<double>(new[] { 1, 1 });
                token[0, 0] = next[step];
                logits.AddRange(model.PredictWithContext(token, new InferenceForwardContext(sequence, 3 + step)).ToArray());
            }
        }
        finally
        {
            model.ShutdownRanks();
        }

        return logits;
    }

    [Theory]
    [InlineData(2, null)]
    [InlineData(2, 2)]
    public void ShardedLogits_MatchTheUnshardedModel(int worldSize, int? numKVHeads)
    {
        var reference = Logits(1, numKVHeads);
        var sharded = Logits(worldSize, numKVHeads);

        Assert.Equal(reference.Count, sharded.Count);
        Assert.Equal((3 + 3) * Vocab, reference.Count);
        for (int i = 0; i < reference.Count; i++)
            Assert.Equal(reference[i], sharded[i], 9);
    }

    [Fact]
    public void ShardCounts_ThatDoNotDivideTheHeadsOrFfn_AreRejected()
    {
        Assert.Throws<ArgumentException>(() =>
            new TensorParallelPagedModel<double>(3, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab));
        Assert.Throws<ArgumentException>(() =>
            new TensorParallelPagedModel<double>(2, EmbedDim, NumHeads, NumLayers, 15, Vocab));
    }
}
