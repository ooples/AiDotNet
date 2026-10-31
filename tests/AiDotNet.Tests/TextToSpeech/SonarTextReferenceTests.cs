using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SONAR's text encoder against transformers' M2M100Encoder (<c>tools/reference-data/sonar_text_reference.py</c>): a tiny
/// seeded encoder's states and embeddings; the released encoder and NLLB tokenizer when SONAR_TEXT_ENCODER names the
/// cointegrated/SONAR_200_text_encoder directory.
/// </summary>
public class SonarTextReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string name = "sonar_text_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    });

    private static Dictionary<string, double[]> State(string base64)
    {
        var bytes = Convert.FromBase64String(base64);
        long headerLength = BitConverter.ToInt64(bytes, 0);
        var header = JObject.Parse(Encoding.UTF8.GetString(bytes, 8, (int)headerLength));
        var result = new Dictionary<string, double[]>();
        int start = 8 + (int)headerLength;
        foreach (var entry in header.Properties())
        {
            if (entry.Name == "__metadata__" || (string)entry.Value["dtype"]! != "F64") continue;
            var offsets = entry.Value["data_offsets"]!.Select(v => (int)v).ToArray();
            result[entry.Name] = Enumerable.Range(0, (offsets[1] - offsets[0]) / 8).Select(i => BitConverter.ToDouble(bytes, start + offsets[0] + 8 * i)).ToArray();
        }
        return result;
    }

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(values[i])), $"{what}[{i}]: {values[i]:R} vs {actual[i]:R}");
    }

    [Fact(Timeout = 60000)]
    public async Task TinyEncoder_MatchesTransformers()
    {
        await Task.Yield();
        var f = Data.Value;
        var c = f["config"]!;
        var encoder = new SonarTextEncoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new SonarTextEncoderConfiguration
        {
            VocabSize = (int)c["vocab"]!, Width = (int)c["width"]!, Layers = (int)c["layers"]!, Heads = (int)c["heads"]!, FeedForward = (int)c["ffn"]!,
        });
        var state = State((string)f["state"]!);
        encoder.LoadTorchWeights(name => state.TryGetValue(name, out var v) ? v : null);
        var ids = f["ids"]!.Select(t => t.Select(v => (int)v).ToArray()).ToArray();
        using var _ = new NoGradScope<double>();
        for (int i = 0; i < ids.Length; i++)
        {
            Close(f[$"states_{i}"]!, encoder.Forward(ids[i]), $"states {i}", 1e-10);
            Close(f[$"embedding_{i}"]!, encoder.Embed(ids[i]), $"embedding {i}", 1e-10);
        }
    }

    [Fact(Timeout = 1800000)]
    public async Task RealEncoderAndTokenizer_MatchTransformers()
    {
        await Task.Yield();
        if (Environment.GetEnvironmentVariable("SONAR_TEXT_ENCODER") is not { Length: > 0 } dir || !Directory.Exists(dir)) return;
        var tokenizer = SonarTextTokenizer.FromDirectory(dir);
        var cases = Data.Value["real"]!.ToArray();
        foreach (var c in cases)
            Assert.Equal(c["ids"]!.Select(v => (int)v).ToArray(), tokenizer.Encode((string)c["text"]!, (string)c["language"]!));
        var encoder = new SonarTextEncoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new SonarTextEncoderConfiguration());
        var weights = new WeightLoader().LoadWeights(Path.Combine(dir, "model.safetensors"));
        encoder.LoadTorchWeights(name => weights.TryGetValue(name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null);
        using var _ = new NoGradScope<double>();
        foreach (var c in cases)
            Close(c["embedding"]!, encoder.Embed(c["ids"]!.Select(v => (int)v).ToArray()), (string)c["text"]!, 1e-8);
    }
}
