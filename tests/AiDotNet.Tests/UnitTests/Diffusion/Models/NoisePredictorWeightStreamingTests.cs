using System;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Regression coverage for diffusion predictors that used lazy layer
/// construction but were not wired into the transparent streaming allocator.
/// </summary>
[Collection("WeightStreaming-Singleton")]
public sealed class NoisePredictorWeightStreamingTests
{
    private const long ResidentCap = 16 * 1024 * 1024;

    [Fact]
    public void UNetPredictNoise_ForcedStreaming_RegistersResolvedWeights()
    {
        using var _ = ForceStreamingForTinyModel();

        using var predictor = new UNetNoisePredictor<float>(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocks: 1,
            attentionResolutions: [],
            contextDim: 16,
            numHeads: 1,
            inputHeight: 8,
            seed: 42);

        var input = new Tensor<float>(new[] { 1, 4, 8, 8 });
        var output = predictor.PredictNoise(input, timestep: 25);

        Assert.Equal(input.Length, output.Length);
        AssertStreamingRegistered();
    }

    [Fact]
    public void MMDiTPredictNoise_ForcedStreaming_RegistersResolvedWeights()
    {
        using var _ = ForceStreamingForTinyModel();

        using var predictor = new MMDiTNoisePredictor<float>(
            inputChannels: 4,
            hiddenSize: 32,
            numJointLayers: 1,
            numSingleLayers: 0,
            numHeads: 4,
            patchSize: 2,
            contextDim: 16,
            seed: 42);

        var input = new Tensor<float>(new[] { 1, 4, 8, 8 });
        var conditioning = new Tensor<float>(new[] { 1, 1, 16 });
        var output = predictor.PredictNoise(input, timestep: 25, conditioning);

        Assert.Equal(input.Length, output.Length);
        AssertStreamingRegistered();
    }

    [Fact]
    public void UViTPredictNoise_ForcedStreaming_RegistersResolvedWeights()
    {
        using var _ = ForceStreamingForTinyModel();

        using var predictor = new UViTNoisePredictor<float>(
            inputChannels: 4,
            hiddenSize: 32,
            numLayers: 2,
            numHeads: 4,
            patchSize: 2,
            contextDim: 0,
            latentSpatialSize: 8,
            seed: 42);

        var input = new Tensor<float>(new[] { 1, 4, 8, 8 });
        var output = predictor.PredictNoise(input, timestep: 25);

        Assert.Equal(input.Length, output.Length);
        AssertStreamingRegistered();
    }

    [Fact]
    public void PredictNoise_Dispose_ReleasesRegisteredStreamingWeights()
    {
        using var _ = ForceStreamingForTinyModel();

        var predictor = new UViTNoisePredictor<float>(
            inputChannels: 4,
            hiddenSize: 32,
            numLayers: 2,
            numHeads: 4,
            patchSize: 2,
            contextDim: 0,
            latentSpatialSize: 8,
            seed: 42);

        var input = new Tensor<float>(new[] { 1, 4, 8, 8 });
        var output = predictor.PredictNoise(input, timestep: 25);

        Assert.Equal(input.Length, output.Length);
        AssertStreamingRegistered();

        predictor.Dispose();

        var report = WeightRegistry.GetStreamingReport();
        Assert.Equal(0, report.RegisteredEntryCount);
    }

    [Fact]
    public void PredictNoise_WithExistingRegisteredStreamingWeights_JoinsActiveStreamingPool()
    {
        using var _ = ForceStreamingForTinyModel();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = ResidentCap,
            TransparentAutoEviction = true,
        });

        var external = WeightRegistry.AllocateStreaming<float>(new[] { 128, 128 });
        WeightRegistry.RegisterWeight(external);

        try
        {
            var before = WeightRegistry.GetStreamingReport().RegisteredEntryCount;

            using var predictor = new UViTNoisePredictor<float>(
                inputChannels: 4,
                hiddenSize: 32,
                numLayers: 2,
                numHeads: 4,
                patchSize: 2,
                contextDim: 0,
                latentSpatialSize: 8,
                seed: 42);

            var input = new Tensor<float>(new[] { 1, 4, 8, 8 });
            var output = predictor.PredictNoise(input, timestep: 25);

            Assert.Equal(input.Length, output.Length);
            Assert.True(WeightRegistry.GetStreamingReport().RegisteredEntryCount > before,
                "A predictor entering an active transparent pool should still register its resolved weights.");
        }
        finally
        {
            WeightRegistry.UnregisterWeight(external);
        }
    }

    [Fact]
    public void PredictNoise_WithPendingStreamingReservation_JoinsActiveStreamingPool()
    {
        using var _ = ForceStreamingForTinyModel();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = ResidentCap,
            TransparentAutoEviction = true,
        });

        var pending = WeightRegistry.AllocateStreaming<float>(new[] { 128, 128 });

        try
        {
            using var predictor = new UViTNoisePredictor<float>(
                inputChannels: 4,
                hiddenSize: 32,
                numLayers: 2,
                numHeads: 4,
                patchSize: 2,
                contextDim: 0,
                latentSpatialSize: 8,
                seed: 42);

            var input = new Tensor<float>(new[] { 1, 4, 8, 8 });
            var output = predictor.PredictNoise(input, timestep: 25);

            Assert.Equal(input.Length, output.Length);
            AssertStreamingRegistered();
        }
        finally
        {
            WeightRegistry.UnregisterWeight(pending);
        }
    }

    private static IDisposable ForceStreamingForTinyModel()
    {
        WeightRegistry.Reset();
        NoisePredictorBase<float>.StreamingThresholdOverride = 1;
        NoisePredictorBase<float>.StreamingResidentCapOverride = ResidentCap;
        return new StreamingOverrideScope();
    }

    private static void AssertStreamingRegistered()
    {
        var report = WeightRegistry.GetStreamingReport();
        Assert.True(report.RegisteredEntryCount > 0,
            "Forced streaming should register resolved lazy weights after the first forward.");
    }

    private sealed class StreamingOverrideScope : IDisposable
    {
        public void Dispose()
        {
            NoisePredictorBase<float>.StreamingThresholdOverride = null;
            NoisePredictorBase<float>.StreamingResidentCapOverride = null;
            WeightRegistry.Reset();
        }
    }
}
