using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

internal static class StableDiffusionFamilyTestScaffold
{
    internal static UNetNoisePredictor<T> CreateUnet<T>(
        int inputChannels = 4,
        int outputChannels = 4,
        int inputHeight = 8,
        int contextDim = 0,
        int seed = 42)
        => new(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocks: 1,
            attentionResolutions: [0],
            contextDim: contextDim,
            numHeads: 1,
            inputHeight: inputHeight,
            seed: seed);

    internal static StandardVAE<T> CreateVae<T>(
        int latentChannels = 4,
        double latentScaleFactor = 0.18215,
        int seed = 42)
        => new(
            inputChannels: 3,
            latentChannels: latentChannels,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocksPerLevel: 1,
            latentScaleFactor: latentScaleFactor,
            seed: seed);

    internal static SiTPredictor<T> CreateSiT<T>(
        int inputChannels = 16,
        int hiddenSize = 16,
        int numLayers = 1,
        int numHeads = 2,
        int seed = 42)
        => new(
            inputChannels: inputChannels,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numHeads: numHeads,
            seed: seed);

    internal static MMDiTXNoisePredictor<T> CreateMMDiTX<T>(
        int inputChannels = 16,
        int contextDim = 4096,
        int hiddenSize = 16,
        int numJointLayers = 1,
        int numHeads = 2,
        int seed = 42)
        => new(
            inputChannels: inputChannels,
            contextDim: contextDim,
            seed: seed,
            hiddenSize: hiddenSize,
            numJointLayers: numJointLayers,
            numHeads: numHeads);

    internal static EMMDiTPredictor<T> CreateEMMDiT<T>(
        int inputChannels = 32,
        int contextDim = 4096,
        int hiddenSize = 16,
        int numLayers = 1,
        int seed = 42)
        => new(
            inputChannels: inputChannels,
            contextDim: contextDim,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            seed: seed);

    internal static FlagDiTPredictor<T> CreateFlagDiT<T>(
        int inputChannels = 16,
        int contextDim = 4096,
        int hiddenSize = 16,
        int numLayers = 1,
        int numHeads = 2,
        int seed = 42)
        => new(
            inputChannels: inputChannels,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numHeads: numHeads,
            contextDim: contextDim,
            seed: seed);

    internal static DeepCompressionVAE<T> CreateDeepCompressionVae<T>(
        int latentChannels = 32,
        int downsampleFactor = 2,
        int seed = 42)
        => new(
            inputChannels: 3,
            latentChannels: latentChannels,
            downsampleFactor: downsampleFactor,
            baseChannels: 8,
            seed: seed);

    internal static DiffusionModelOptions<T> CreateOneStepOptions<T>(
        double betaStart = 0.00085,
        double betaEnd = 0.012,
        BetaSchedule betaSchedule = BetaSchedule.ScaledLinear,
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
