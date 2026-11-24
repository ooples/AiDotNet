using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A model that trains only on <see cref="TtsTrainingSample{T}"/> supervision must refuse a token/mel pair with the
/// NotSupportedException that names the entry point, as every other such model does. XTTSv2, E3-TTS, DiTTo-TTS and
/// MetaVoice-1B instead fell through to the base step, which walked their layers on bare token ids; the model
/// performance census (#2390) failed on all four with shape errors ("Conv1DLayer requires rank-3 [B, C, T] input;
/// got [[4]]", "Expected last dimension 1, got [[4]]", "... got [[4, 8]]", "TiedEmbeddingLayer`1.input requires
/// token indices in [0, 128)").
/// </summary>
public sealed class SampleObjectiveTokenMelTrainingTests
{
    public static TheoryData<string> Models => new() { "XTTSv2", "E3TTS", "DiTToTTS", "MetaVoice1B" };

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Models))]
    public async Task TokenMelTrain_OnSampleObjectiveModel_ThrowsNotSupportedNamingTheSampleEntryPoint(string name)
    {
        await Task.Yield();
        using var model = Create(name);
        Assert.True(model.TrainsOnSampleObjective);
        var tokens = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = i + 1;
        var mel = new Tensor<double>(new[] { 1, 4, 8 });

        var error = Assert.Throws<NotSupportedException>(() => model.Train(tokens, mel));
        Assert.Contains("Train(TtsTrainingSample)", error.Message);
    }

    private static TtsModelBase<double> Create(string name)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 16, outputSize: 12);
        return name switch
        {
            "XTTSv2" => new XTTSv2<double>(architecture, XTTSv2PaperTests.SmallOptions()),
            "E3TTS" => new E3TTS<double>(architecture, E3TTSPaperTests.TinyOptions()),
            "DiTToTTS" => new DiTToTTS<double>(architecture, DiTToTTSPaperTests.TinyOptions()),
            "MetaVoice1B" => MetaVoicePublicPipelineTests.Model(),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
        };
    }
}
