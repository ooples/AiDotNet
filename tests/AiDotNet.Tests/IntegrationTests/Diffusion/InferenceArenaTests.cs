using System;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Eager inference forwards run inside a reused TensorArena from the second forward of an input shape on
/// (ooples/AiDotNet.Tensors#653). The arena only changes where intermediates live: every prediction must equal the
/// arena-free prediction bit for bit, including after shape changes and repeated inputs, and a returned prediction
/// must not change when later predictions reuse the arena.
/// </summary>
[Collection("InferenceArenaToggle")]
public class InferenceArenaTests
{
    public static TheoryData<string> Predictors => new() { "DiT", "UNet" };

    private static NoisePredictorBase<float> Create(string name) => name switch
    {
        "DiT" => new DiTNoisePredictor<float>(inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 2,
            patchSize: 2, contextDim: 16, latentSpatialSize: 8, seed: 42),
        "UNet" => new UNetNoisePredictor<float>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
            channelMultipliers: [1, 2], numResBlocks: 1, attentionResolutions: [1], contextDim: 0, numHeads: 2,
            inputHeight: 8, seed: 42),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    private static Tensor<float> Sample(int batch, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var x = new Tensor<float>([batch, 4, 8, 8]);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    // (batch, input seed, timestep): repeats, a timestep change, and a batch-shape change and back.
    private static readonly (int Batch, int Seed, int T)[] Calls =
    [
        (1, 1, 3), (1, 2, 500), (1, 1, 3), (2, 3, 250), (1, 2, 500), (2, 4, 999), (1, 1, 3), (2, 3, 250),
    ];

    private static float[][] Run(NoisePredictorBase<float> predictor, out Tensor<float>[] kept)
    {
        kept = new Tensor<float>[Calls.Length];
        var copies = new float[Calls.Length][];
        for (int i = 0; i < Calls.Length; i++)
        {
            var (batch, seed, t) = Calls[i];
            kept[i] = predictor.PredictNoise(Sample(batch, seed), t);
            copies[i] = kept[i].ToArray();
        }
        return copies;
    }

    private static void AssertUnchanged(string label, Tensor<float>[] kept, float[][] whenReturned)
    {
        for (int i = 0; i < kept.Length; i++)
        {
            var now = kept[i].ToArray();
            for (int j = 0; j < now.Length; j++)
                Assert.True(AiDotNet.MixedPrecision.BitConverterHelper.SingleToInt32Bits(now[j]) == AiDotNet.MixedPrecision.BitConverterHelper.SingleToInt32Bits(whenReturned[i][j]),
                    $"{label}: call {i} element {j} changed after later predictions: {now[j]} vs {whenReturned[i][j]}");
        }
    }

    [Theory]
    [MemberData(nameof(Predictors))]
    public void ArenaPredictions_MatchArenaFree_BitForBit(string name)
    {
        bool saved = NoisePredictorBase<float>.InferenceArenaEnabled;
        try
        {
            NoisePredictorBase<float>.InferenceArenaEnabled = false;
            var reference = Run(Create(name), out var keptReference);
            AssertUnchanged(name + " without the arena", keptReference, reference);

            NoisePredictorBase<float>.InferenceArenaEnabled = true;
            var predictor = Create(name);
            var withArena = Run(predictor, out var kept);

            // Eight calls over two input shapes; the first of each shape runs outside the arena.
            Assert.Equal(6L, predictor.InferenceArenaForwards);
            for (int i = 0; i < Calls.Length; i++)
            {
                Assert.Equal(reference[i].Length, withArena[i].Length);
                for (int j = 0; j < reference[i].Length; j++)
                    Assert.True(AiDotNet.MixedPrecision.BitConverterHelper.SingleToInt32Bits(reference[i][j]) == AiDotNet.MixedPrecision.BitConverterHelper.SingleToInt32Bits(withArena[i][j]),
                        $"{name} call {i} element {j}: {withArena[i][j]} with the arena vs {reference[i][j]} without");
            }
            // A returned prediction still holds its values after every later call reused the arena.
            AssertUnchanged(name + " with the arena", kept, withArena);
        }
        finally
        {
            NoisePredictorBase<float>.InferenceArenaEnabled = saved;
        }
    }

    /// <summary>
    /// Arenas are rented per forward from a bounded pool, not kept per thread: one thread after another predicting
    /// leaves one idle arena, where a per-thread arena kept one for every thread that had ever called, exited or not.
    /// </summary>
    [Theory]
    [MemberData(nameof(Predictors))]
    public void ShortLivedThreads_LeaveOneIdleArena_NotOnePerThread(string name)
    {
        bool saved = NoisePredictorBase<float>.InferenceArenaEnabled;
        try
        {
            NoisePredictorBase<float>.InferenceArenaEnabled = true;
            var predictor = Create(name);
            predictor.PredictNoise(Sample(1, 1), 3);
            for (int i = 0; i < 6; i++)
            {
                int inputSeed = 2 + i;
                // Join does not carry an exception back from a raw Thread, so a throw inside the worker would go
                // unhandled there and take the test host down instead of failing this theory case.
                Exception? workerFailure = null;
                var worker = new System.Threading.Thread(() =>
                {
                    try
                    {
                        predictor.PredictNoise(Sample(1, inputSeed), 500);
                    }
                    catch (Exception ex)
                    {
                        workerFailure = ex;
                    }
                });
                worker.Start();
                worker.Join();
                Assert.True(workerFailure is null, $"{name}: worker {i} threw {workerFailure}");
            }

            Assert.Equal(6L, predictor.InferenceArenaForwards);
            Assert.Equal(1, predictor.IdleInferenceArenaCount);
        }
        finally
        {
            NoisePredictorBase<float>.InferenceArenaEnabled = saved;
        }
    }
}