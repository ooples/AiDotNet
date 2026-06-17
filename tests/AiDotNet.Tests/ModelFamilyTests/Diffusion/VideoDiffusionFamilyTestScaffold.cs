using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

internal static class VideoDiffusionFamilyTestScaffold
{
    internal static VideoUNetPredictor<T> CreateVideoUnet<T>(
        int inputHeight = 8,
        int inputWidth = 8,
        int contextDim = 0,
        bool supportsImageConditioning = true,
        int seed = 42)
        => new(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocks: 1,
            attentionResolutions: [0],
            numTemporalLayers: 1,
            contextDim: contextDim,
            numHeads: 1,
            supportsImageConditioning: supportsImageConditioning,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            numFrames: 4,
            seed: seed);

    internal static TemporalVAE<T> CreateTemporalVae<T>(
        int latentChannels = 4,
        double latentScaleFactor = 0.18215,
        bool causalMode = false,
        int seed = 42)
        => new(
            inputChannels: 3,
            latentChannels: latentChannels,
            baseChannels: 8,
            channelMultipliers: [1],
            numTemporalLayers: 1,
            temporalKernelSize: 3,
            causalMode: causalMode,
            latentScaleFactor: latentScaleFactor,
            seed: seed);

    internal static StandardVAE<T> CreateVae<T>(
        double latentScaleFactor = 0.18215,
        int seed = 42)
        => new(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocksPerLevel: 1,
            latentScaleFactor: latentScaleFactor,
            seed: seed);

    internal static AudioVAE<T> CreateAudioVae<T>(
        int melChannels = 128,
        int latentChannels = 64,
        int seed = 42)
        => new(
            melChannels: melChannels,
            latentChannels: latentChannels,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocks: 1,
            seed: seed);

    internal static DiTNoisePredictor<T> CreateDiT<T>(
        int inputChannels = 4,
        int latentSpatialSize = 8,
        int contextDim = 4096,
        int patchSize = 2,
        int seed = 42)
        => new(
            inputChannels: inputChannels,
            hiddenSize: 16,
            numLayers: 1,
            numHeads: 2,
            patchSize: patchSize,
            contextDim: contextDim,
            latentSpatialSize: latentSpatialSize,
            seed: seed);

    internal static DiffusionModelOptions<T> CreateOneStepOptions<T>(
        double betaStart = 0.0001,
        double betaEnd = 0.02,
        BetaSchedule betaSchedule = BetaSchedule.Linear,
        double learningRate = 0.00001,
        int seed = 42)
        => new()
        {
            TrainTimesteps = 1000,
            BetaStart = betaStart,
            BetaEnd = betaEnd,
            BetaSchedule = betaSchedule,
            LearningRate = learningRate,
            Seed = seed,
            DefaultInferenceSteps = 1
        };
}
