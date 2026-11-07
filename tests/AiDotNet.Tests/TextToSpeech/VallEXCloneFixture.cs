using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>The paper tests' bounded native bilingual AR/NAR and 75 Hz EnCodec cloning facade.</summary>
internal static class VallEXCloneFixture
{
    internal static VALLEXCloneOptions Options(string referenceText = "a prompt") => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, FeedForwardDim = 32,
        TextTokens = 192, NumCodebooks = 3, CodebookSize = 64, MaxCodesPerTextToken = 2,
        LearningRate = 3e-3, WarmupSteps = 0, DropoutRate = 0, MinReferenceDurationSec = 0.01,
        ReferenceText = referenceText,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 3, CodebookSize = 64, Filters = 2,
            Ratios = [8, 5, 4, 2], Dimension = 8, ResidualKernelSizes = [3, 1],
            TargetBandwidthKbps = 1.35, TargetBandwidths = [1.35], SegmentSize = 640,
            IncludeDiscriminators = false, KMeansIterations = 0, DeadCodeThreshold = 0, LstmLayers = 1,
        }
    };

    internal static VALLEXClone<T> Create<T>() =>
        new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, Options());
}
