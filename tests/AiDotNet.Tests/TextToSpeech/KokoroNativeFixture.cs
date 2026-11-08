using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Full Kokoro-82M topology with synthetic style; original weights are audited separately.</summary>
internal static class KokoroNativeFixture
{
    internal static Kokoro<T> Create<T>()
    {
        var operations = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var model = new Kokoro<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 1200),
            new KokoroOptions { UseNativeEnglishFallback = true, Speed = 20, LearningRate = 1e-5, Seed = 2294, SynthesisSeed = 2294 });
        model.LoadAuthorConfiguration(Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "kokoro_author_config.json"));
        model.Voice = new TtsVoice<T>
        {
            Reference = new Tensor<T>(new[] { 1, 256 }, new Vector<T>(Enumerable.Range(0, 256)
                .Select(index => operations.FromDouble(.03 * Math.Sin(index * .13))).ToArray()))
        };
        return model;
    }
}
