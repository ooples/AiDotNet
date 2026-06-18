using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// G4 (#1624) activation checkpointing on the diffusion predictors. Checkpointing trades compute for
/// memory by recomputing block activations in backward instead of storing them — it is mathematically
/// transparent, so the forward result MUST be identical whether or not it is engaged. (The memory
/// reduction itself only manifests at foundation scale during training and is verified in CI; here we
/// pin the correctness contract on a tiny variant.)
/// </summary>
public class ActivationCheckpointingTests
{
    private static FlagDiTPredictor<double> TinyFlagDiT(int seed) =>
        new FlagDiTPredictor<double>(
            inputChannels: 4, hiddenSize: 32, numLayers: 3,
            numHeads: 4, numKVHeads: 2, contextDim: 32, latentSize: 8, seed: seed);

    private static Tensor<double> Input()
    {
        var t = new Tensor<double>(new[] { 1, 4, 8, 8 });
        for (int i = 0; i < t.Length; i++) t[i] = (i % 11) * 0.02 - 0.1;
        return t;
    }

    private static void AssertForwardTransparent(NoisePredictorBase<double> predictor, Tensor<double> input)
    {
        predictor.ActivationCheckpointingEnabled = false;
        var eager = predictor.PredictNoise(input, timestep: 0);

        predictor.ActivationCheckpointingEnabled = true;
        var checkpointed = predictor.PredictNoise(input, timestep: 0);

        Assert.Equal(eager.Length, checkpointed.Length);
        for (int i = 0; i < eager.Length; i++)
            Assert.Equal(eager[i], checkpointed[i], 10);
    }

    [Fact]
    public void FlagDiT_Checkpointing_IsForwardTransparent()
        => AssertForwardTransparent(TinyFlagDiT(seed: 7), Input());

    [Fact]
    public void MMDiT_Checkpointing_IsForwardTransparent()
        => AssertForwardTransparent(
            new MMDiTNoisePredictor<double>(
                inputChannels: 4, hiddenSize: 32, numJointLayers: 2, numSingleLayers: 1,
                numHeads: 4, patchSize: 2, contextDim: 32, seed: 7),
            Input());

    [Fact]
    public void UViT_Checkpointing_IsForwardTransparent()
        => AssertForwardTransparent(
            new UViTNoisePredictor<double>(
                inputChannels: 4, hiddenSize: 32, numLayers: 2, numHeads: 4,
                patchSize: 2, contextDim: 0, latentSpatialSize: 8, seed: 7),
            Input());

    [Fact]
    public void ActivationCheckpointing_AutoEngages_AboveThreshold()
    {
        var predictor = TinyFlagDiT(seed: 1); // tiny → well below the default 100M threshold

        // Default (auto): tiny model stays eager.
        Assert.False(predictor.ActivationCheckpointingEnabled);

        // Lower the threshold below the tiny model's size → auto-engages.
        var prev = NoisePredictorBase<double>.CheckpointingThresholdOverride;
        try
        {
            NoisePredictorBase<double>.CheckpointingThresholdOverride = 0;
            var auto = TinyFlagDiT(seed: 1);
            Assert.True(auto.ActivationCheckpointingEnabled);
        }
        finally
        {
            NoisePredictorBase<double>.CheckpointingThresholdOverride = prev;
        }

        // Explicit override beats the threshold.
        predictor.ActivationCheckpointingEnabled = true;
        Assert.True(predictor.ActivationCheckpointingEnabled);
    }
}
