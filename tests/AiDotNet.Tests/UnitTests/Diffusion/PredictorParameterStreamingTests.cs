using System.Collections.Generic;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Verifies the #1624 streaming parameter API (GetParameterChunks / SetParameterChunks) on
/// foundation-scale DiT predictors using TINY variants so the invariants are checked without the
/// multi-billion-parameter allocation that OOMs at default size. The load-bearing contract is
/// PER-INDEX correspondence: the flat concatenation of GetParameterChunks must equal GetParameters
/// element-for-element (the optimizer reconstructs per-tensor gradient deltas in chunk order), and
/// SetParameterChunks must apply a chunk stream back exactly.
/// </summary>
public class PredictorParameterStreamingTests
{
    private static FlagDiTPredictor<double> TinyFlagDiT(int seed) =>
        new FlagDiTPredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 2,
            numHeads: 4, numKVHeads: 2, contextDim: 32, latentSize: 8, seed: seed);

    [Fact]
    public void FlagDiT_GetParameterChunks_AreIndexIdenticalToGetParameters()
    {
        var predictor = TinyFlagDiT(seed: 7);

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

    [Fact]
    public void FlagDiT_SetParameterChunks_AppliesAStreamedSourceExactly()
    {
        var source = TinyFlagDiT(seed: 1);
        var dest = TinyFlagDiT(seed: 2);

        var sourceFlat = source.GetParameters();
        var destBefore = dest.GetParameters();

        // Sanity: different seeds really do start with different weights, so the round-trip below
        // is a genuine assignment and not a no-op.
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
