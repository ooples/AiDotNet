using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice's networks against the official FunAudioLLM/CosyVoice modules in float64
/// (tools/reference-data/cosyvoice_reference.py): the same weights, the same inputs, the loss and the weights after one
/// SGD step.
/// </summary>
public class CosyVoiceReferenceTests
{
    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "cosyvoice_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static Dictionary<string, double[]> StateDict(JsonElement e)
        => e.EnumerateObject().ToDictionary(p => p.Name, p => Values(p.Value));

    private static CosyVoiceLanguageModel<double> LanguageModel(IEngine engine, List<LayerBase<double>> layers, JsonElement c)
    {
        int width = c.GetProperty("width").GetInt32(), heads = c.GetProperty("heads").GetInt32();
        int units = c.GetProperty("units").GetInt32(), blocks = c.GetProperty("blocks").GetInt32();
        int textIn = c.GetProperty("text_in").GetInt32(), textOut = c.GetProperty("text_out").GetInt32();
        return new CosyVoiceLanguageModel<double>(engine, layers, new CosyVoiceLanguageModelConfiguration(
            TextVocabulary: c.GetProperty("text_vocab").GetInt32(), SpeechTokens: c.GetProperty("speech_tokens").GetInt32(),
            TextEncoderInput: textIn,
            TextEncoder: new WenetEncoderConfiguration(textIn, textOut, heads, units, blocks, Dropout: 0, PositionalDropout: 0,
                InputLayer: WenetInputLayer.Linear, SwishActivation: true, StaticChunkSize: 1),
            Language: new WenetEncoderConfiguration(width, width, heads, units, blocks, Dropout: 0, PositionalDropout: 0,
                InputLayer: WenetInputLayer.LinearLegacy, SwishActivation: false, StaticChunkSize: 1),
            SpeakerEmbeddingDim: c.GetProperty("spk").GetInt32()));
    }

    [Fact(Timeout = 120000)]
    public async Task LanguageModel_LossAndOneSgdStep_MatchTheOfficialCode()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value.GetProperty("language_model");
        var c = k.GetProperty("config");
        var layers = new List<LayerBase<double>>();
        var model = LanguageModel(engine, layers, c);
        model.LoadTorch(StateDict(k.GetProperty("initial")));
        var text = Ints(k.GetProperty("text"));
        var speech = Ints(k.GetProperty("speech"));
        var embedding = Values(k.GetProperty("embedding"));

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        double loss;
        using (var tape = new GradientTape<double>())
        {
            var (sum, count) = model.Loss(text, speech, embedding, training: true, new Random(0));
            var mean = engine.TensorMultiplyScalar(sum, 1.0 / count);
            loss = mean[0];
            grads = tape.ComputeGradients(mean, sources: parameters);
        }
        double expected = k.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - expected) < 1e-9, $"loss {loss} vs {expected}");

        double lr = k.GetProperty("lr").GetDouble();
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
            engine.InvalidatePersistentTensor(p);
        }

        var expectedLayers = new List<LayerBase<double>>();
        var reference = LanguageModel(engine, expectedLayers, c);
        reference.LoadTorch(StateDict(k.GetProperty("final")));
        Assert.Equal(expectedLayers.Count, layers.Count);
        for (int l = 0; l < layers.Count; l++)
        {
            var a = layers[l].GetParameters();
            var e = expectedLayers[l].GetParameters();
            Assert.Equal(e.Length, a.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - a[i]) < 1e-9, $"layer {l} ({layers[l].GetType().Name}) parameter {i}: {e[i]} vs {a[i]}");
        }
    }
}
