using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>A bounded complete native Qwen2/DiT/CausalHiFT/CAM++ model with the release's tokenizer vocabulary.</summary>
internal static class CosyVoice3Fixture
{
    internal static CosyVoice3Options Options() => new()
    {
        LLMDim = 16, NumLLMLayers = 1, NumHeads = 2, KeyValueHeads = 1, LmFeedForwardDim = 32,
        ReservedSpeechTokens = 4, MelChannels = 8, SpeakerEmbeddingDim = 8,
        HiftBaseChannels = 16, HiftHarmonics = 1, HiftUpsampleRates = new[] { 2 },
        HiftUpsampleKernels = new[] { 4 }, HiftSourceKernels = new[] { 3 }, HopSize = 8, FftSize = 32,
        DiTInputDim = 8, PreLookaheadChannels = 16, DiTDim = 16, DiTDepth = 1,
        DiTHeads = 2, DiTHeadDim = 8, DiTFeedForwardMultiplier = 2, DiTTimesteps = 2,
        SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 4, IncludeDiscriminators = false,
        DropoutRate = 0, MaxTextLength = 32, VocoderSegmentLength = 32,
    };

    internal static CosyVoice3<double> Create(CosyVoice3Options options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);
}
