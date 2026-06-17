using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// Shared FLUX-family components for model-family invariant tests.
/// </summary>
internal static class FluxFamilyTestScaffold
{
    internal const int LatentChannels = 16;
    internal const int ContextDim = 4096;

    internal static FluxDoubleStreamPredictor<T> CreatePredictor<T>(
        FluxPredictorVariant variant = FluxPredictorVariant.Dev,
        int? seed = 42)
        => new(
            variant: variant,
            inputChannels: LatentChannels,
            contextDim: ContextDim,
            seed: seed,
            hiddenSize: 64,
            numJointLayers: 2,
            numSingleLayers: 2);

    internal static StandardVAE<T> CreateVae<T>(
        double latentScaleFactor = 1.0,
        int? seed = 42)
        => new(
            inputChannels: 3,
            latentChannels: LatentChannels,
            baseChannels: 16,
            channelMultipliers: [1, 2],
            numResBlocksPerLevel: 1,
            latentScaleFactor: latentScaleFactor,
            seed: seed);

    internal static DiffusionModelOptions<T> CreateOneStepOptions<T>(
        double betaStart = 0.0001,
        double betaEnd = 0.02,
        BetaSchedule betaSchedule = BetaSchedule.Linear)
        => new()
        {
            TrainTimesteps = 1000,
            BetaStart = betaStart,
            BetaEnd = betaEnd,
            BetaSchedule = betaSchedule,
            Seed = 42,
            DefaultInferenceSteps = 1,
        };
}
