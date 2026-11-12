using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Native V2 converter, complete training graph and separate native MeloTTS base voice.</summary>
internal static class OpenVoiceNativeFixture
{
    internal static OpenVoice<T> VersionOne<T>()
    {
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1024, outputSize: 1024) { RandomSeed = 2294 };
        return new OpenVoice<T>(architecture, new OpenVoiceOptions
        {
            HiddenDim = 4, InterChannels = 4, SpeakerEmbeddingDim = 3, FftSize = 512,
            Training = new OpenVoiceTrainingOptions
            {
                IncludeTrainingComponents = true, IpaVocabularySize = 17, IpaFilterChannels = 8,
                IpaHeads = 1, IpaLayers = 1, IpaDropout = 0, SegmentFrames = 4,
                MelChannels = 4, DiscriminatorWidthDivisor = 64
            }
        });
    }

    internal static OpenVoiceV2<T> VersionTwo<T>()
    {
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1024, outputSize: 1024) { RandomSeed = 2294 };
        return new OpenVoiceV2<T>(architecture, new OpenVoiceV2Options
        {
            HiddenDim = 4, DecoderDim = 4, SpeakerEmbeddingDim = 3, FftSize = 512,
            Training = new OpenVoiceTrainingOptions
            {
                IncludeTrainingComponents = true, IpaVocabularySize = 17, IpaFilterChannels = 8,
                IpaHeads = 1, IpaLayers = 1, IpaDropout = 0, SegmentFrames = 4,
                MelChannels = 4, DiscriminatorWidthDivisor = 64
            }
        })
        {
            BaseSynthesizer = new MeloTTS<T>(architecture, new MeloTTSOptions
            {
                SampleRate = 22050, FftSize = 64, WindowSize = 64, HopSize = 16, MelChannels = 16, SegmentSize = 64,
                HiddenDim = 16, InterChannels = 8, FilterChannels = 32, NumHeads = 2, NumEncoderLayers = 3,
                FlowTransformerLayers = 3, NumFlowSteps = 1, NumSpeakers = 2, SpeakerChannels = 8,
                UpsampleRates = new[] { 4, 4 }, UpsampleKernelSizes = new[] { 8, 8 }, UpsampleInitialChannels = 16,
                ResblockKernelSizes = new[] { 3 }, ResblockDilationSizes = new[] { new[] { 1, 3, 5 } },
                BertChannels = 16, JaBertChannels = 16, BertLayers = 2, BertHeads = 2, BertIntermediate = 32,
                DiscriminatorWidthDivisor = 16
            })
        };
    }
}
