using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>A bounded complete native Qwen2/DiT/CausalHiFT/CAM++ model with the release's tokenizer vocabulary.</summary>
internal static class CosyVoice3Fixture
{
    private sealed class NucleusBoundaryRandom : Random
    {
        public override double NextDouble() => .99;
    }

    internal static int ObserveTopP(CosyVoice3<double> model)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        object? languageModel = null;
        for (var type = model.GetType(); type is not null && languageModel is null; type = type.BaseType)
            languageModel = type.GetProperty("LanguageModel", flags)?.GetValue(model);
        var configuration = languageModel!.GetType().GetProperty("Configuration", flags)!.GetValue(languageModel)!;
        object Setting(string name) => configuration.GetType().GetProperty(name)!.GetValue(configuration)!;
        var sampler = languageModel.GetType().Assembly.GetType("AiDotNet.TextToSpeech.CodecBased.CosyVoiceLm`1")!
            .MakeGenericType(typeof(double));
        var method = sampler.GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)
            .Single(candidate => candidate.Name == "SampleRepetitionAware" && candidate.GetParameters().Length == 8);
        return (int)method.Invoke(null, new object[] { new[] { Math.Log(.6), Math.Log(.4) }, Array.Empty<int>(),
            Setting("TopP"), Setting("TopK"), Setting("RepetitionWindow"), Setting("RepetitionRatio"),
            new NucleusBoundaryRandom(),
            configuration.GetType().GetProperty("ExcludeRepeatedTokenOnFallback")?.GetValue(configuration) ?? true })!;
    }

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
