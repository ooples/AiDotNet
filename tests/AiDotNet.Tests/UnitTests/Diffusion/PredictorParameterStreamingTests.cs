using System.Collections.Generic;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Verifies the #1624 streaming parameter API (GetParameterChunks / SetParameterChunks) on the
/// foundation-scale DiT-family predictors using TINY / small variants so the invariants are checked
/// without the multi-billion-parameter allocation that OOMs at default size. The load-bearing
/// contract is PER-INDEX correspondence: the flat concatenation of GetParameterChunks must equal
/// GetParameters element-for-element (the optimizer reconstructs per-tensor gradient deltas in chunk
/// order), and SetParameterChunks must apply a streamed source back exactly.
/// </summary>
public class PredictorParameterStreamingTests
{
    private static FlagDiTPredictor<double> FlagDiT(int seed) =>
        new FlagDiTPredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 2,
            numHeads: 4, numKVHeads: 2, contextDim: 32, latentSize: 8, seed: seed);

    private static AsymmDiTPredictor<double> AsymmDiT(int seed) =>
        new AsymmDiTPredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 4, contextDim: 32, seed: seed);

    private static SiTPredictor<double> SiT(int seed) =>
        new SiTPredictor<double>(inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 4, seed: seed);

    // EMMDiT has fixed internal dims (1024 hidden, 12 layers ≈ 15M params); small enough to construct.
    private static EMMDiTPredictor<double> EMMDiT(int seed) =>
        new EMMDiTPredictor<double>(inputChannels: 4, contextDim: 64, seed: seed);

    [Fact] public void FlagDiT_Chunks_IndexIdentical() => AssertIndexIdentical(FlagDiT(7));
    [Fact] public void FlagDiT_SetChunks_RoundTrips() => AssertRoundTrips(FlagDiT(1), FlagDiT(2));

    [Fact] public void AsymmDiT_Chunks_IndexIdentical() => AssertIndexIdentical(AsymmDiT(7));
    [Fact] public void AsymmDiT_SetChunks_RoundTrips() => AssertRoundTrips(AsymmDiT(1), AsymmDiT(2));

    [Fact] public void SiT_Chunks_IndexIdentical() => AssertIndexIdentical(SiT(7));
    [Fact] public void SiT_SetChunks_RoundTrips() => AssertRoundTrips(SiT(1), SiT(2));

    [Fact] public void EMMDiT_Chunks_IndexIdentical() => AssertIndexIdentical(EMMDiT(7));
    [Fact] public void EMMDiT_SetChunks_RoundTrips() => AssertRoundTrips(EMMDiT(1), EMMDiT(2));

    private static void AssertIndexIdentical(NoisePredictorBase<double> predictor)
    {
        var flat = predictor.GetParameters();

        long sum = 0;
        var rebuilt = new List<double>();
        foreach (var chunk in predictor.GetParameterChunks())
        {
            var v = chunk.ToVector();
            for (int i = 0; i < v.Length; i++) rebuilt.Add(v[i]);
            sum += chunk.Length;
        }

        Assert.Equal(predictor.ParameterCount, sum);
        Assert.Equal(flat.Length, rebuilt.Count);
        for (int i = 0; i < flat.Length; i++)
            Assert.Equal(flat[i], rebuilt[i], 12);
    }

    private static void AssertRoundTrips(NoisePredictorBase<double> source, NoisePredictorBase<double> dest)
    {
        var sourceFlat = source.GetParameters();
        var destBefore = dest.GetParameters();

        bool anyDifferent = false;
        for (int i = 0; i < sourceFlat.Length && !anyDifferent; i++)
            if (sourceFlat[i] != destBefore[i]) anyDifferent = true;
        Assert.True(anyDifferent, "Test setup invalid: the two seeds produced identical weights.");

        dest.SetParameterChunks(source.GetParameterChunks());

        var destAfter = dest.GetParameters();
        Assert.Equal(sourceFlat.Length, destAfter.Length);
        for (int i = 0; i < sourceFlat.Length; i++)
            Assert.Equal(sourceFlat[i], destAfter[i], 12);
    }
}
