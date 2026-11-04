using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Native Qwen, XLSR and BiCodec, with explicit grammar-constrained smoke sampling.</summary>
internal static class SparkNativeFixture
{
    internal static SparkTTS<T> Create<T>()
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.MaxTextLength = 128;
        options.DropoutRate = 0;
        return new SparkTTS<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 320) { RandomSeed = 2294 }, options);
    }
}
