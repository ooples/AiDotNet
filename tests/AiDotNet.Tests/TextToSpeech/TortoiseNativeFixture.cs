using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>All five native components at framework-test scale; full released sampling is audited separately.</summary>
internal static class TortoiseNativeFixture
{
    internal static TortoiseTTS<T> Create<T>() => new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
        NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 256) { RandomSeed = 2294 }, new TortoiseTTSOptions
    {
        SampleRate = 1600, TokenizerSampleRate = 1600, MelChannels = 8, AutoregressiveMelChannels = 8,
        FftSize = 32, AutoregressiveFftSize = 32, HopSize = 8, AutoregressiveHopSize = 8,
        ConditioningSamples = 64, DiffusionConditioningSamples = 64, VocabSize = 256, MaxTextLength = 128,
        TextStartToken = 255, TextStopToken = 0, CodebookSize = 8, SpeechStartToken = 8, SpeechStopToken = 9,
        LLMDim = 8, TextEncoderDim = 8, NumLLMLayers = 1, NumEncoderLayers = 1, NumHeads = 2,
        AutoregressiveFeedForwardDim = 16, MaxCodecFrames = 32, MaximumAutoregressiveTokens = 4,
        NumAutoregressiveSamples = 2, MinimumAutoregressiveTokens = 1,
        ClvpDim = 8, ClvpLayers = 1, ClvpHeads = 2, ClvpTextLength = 128, ClvpSpeechLength = 64,
        DiffusionDim = 8, DiffusionLayers = 2, DiffusionHeads = 2,
        DiffusionTrainingSteps = 1000, DiffusionInferenceSteps = 2,
        TokenizerLatentDim = 8, TokenizerCodeDim = 2, UnivNetNoiseDim = 4, UnivNetChannels = 4,
        UnivNetUpsampleRates = new[] { 2, 2, 2 }, UnivNetDilations = new[] { 1, 3 },
        UnivNetKernelPredictorHidden = 4, UnivNetPaddingFrames = 1,
        VocoderTraining = new UnivNetOptions
        {
            StftFftSizes = new[] { 16, 32 }, StftHopSizes = new[] { 4, 8 }, StftWindowSizes = new[] { 16, 24 },
            ResolutionDiscriminatorChannels = 2, DiscriminatorPeriods = new[] { 2, 3 },
            PeriodDiscriminatorChannels = new[] { 2, 2, 2, 2, 2 }, PretrainingSteps = 0
        }
    });
}
