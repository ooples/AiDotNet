using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Complete global/local Transformer and three-codebook universal codec at test dimensions.</summary>
internal static class UniAudioNativeFixture
{
    internal static UniAudio<T> Create<T>() => new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
        NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 64) { RandomSeed = 2294 }, new UniAudioOptions
    {
        SampleRate = 320, UseReleasedTextFrontend = false, HopSize = 32, CodecFrameRate = 10, MelChannels = 8,
        NumCodebooks = 3, CodebookSize = 8, CodecTotalQuantizers = 3, CodecFilters = 2, CodecDimension = 6,
        CodecRatios = new[] { 4, 2, 2, 2 }, CodecKMeansIterations = 0, CodecDeadCodeThreshold = 0,
        LLMDim = 12, TextEncoderDim = 12, NumHeads = 3, NumEncoderLayers = 1, NumLLMLayers = 1,
        GlobalFeedForwardDim = 24, LocalFeedForwardDim = 24, JointVocabularySize = 192, VocabSize = 192,
        AudioTokenOffset = 32, PhoneTokenOffset = 80, PhoneVocabularySize = 80,
        MaxTextLength = 128, MaxContextPatches = 256, MaxCodecFrames = 64, PromptSeconds = .1,
        TopK = 4, Temperature = .8, MinimumLengthRatio = 0, MaximumLengthRatio = 1,
        MinimumGeneratedFrames = 1, MaximumGeneratedFrames = 2, SamplingSeed = 7
    });
}
