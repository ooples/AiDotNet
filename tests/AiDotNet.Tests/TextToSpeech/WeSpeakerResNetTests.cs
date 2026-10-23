using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Speaker;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The WeSpeaker ResNet (StyleTTS-ZS's speaker-embedding feature matching) reproduces wenet-e2e/wespeaker: the CLI's
/// mean-normalized Kaldi fbank of the int16-scaled waveform, every stage's feature map and the embedding.
/// </summary>
/// <remarks><c>ReferenceData/wespeaker_reference.json</c> (<c>tools/reference-data/wespeaker_reference.py</c>).</remarks>
public class WeSpeakerResNetTests
{
    private static JObject Reference()
    {
        const string fileName = "wespeaker_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 300000)]
    public async Task ResNet_MatchesTheReference()
    {
        await Task.Yield();
        var fixture = Reference();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var layers = new List<LayerBase<double>>();
        var model = new WeSpeakerResNet<double>(engine, layers, fixture["blocks"]!.Select(v => (int)v).ToArray(), (int)fixture["channels"]!,
            embedding: (int)fixture["embedding_dim"]!);
        var wave = Values(fixture["wave"]!);
        var audio = new Tensor<double>(new[] { wave.Length }, new Vector<double>(wave));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        model.Forward(audio);
        string path = Path.Combine(Path.GetTempPath(), $"wespeaker_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadTorchWeights((name, shape) =>
            {
                Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), weights[name].Length);
                used.Add(name);
                return weights[name].ToVector().Select(v => (double)v).ToArray();
            });
            var unused = weights.Keys.Where(k => !used.Contains(k) && !k.EndsWith("num_batches_tracked", StringComparison.Ordinal)).ToList();
            Assert.True(unused.Count == 0, "Unread reference tensors: " + string.Join(", ", unused));
        }
        finally
        {
            File.Delete(path);
        }

        Close(Values(fixture["features"]!), model.Features(audio), "features");
        var (maps, embedding) = model.Forward(audio);
        var expected = fixture["maps"]!.ToArray();
        Assert.Equal(expected.Length, maps.Count);
        for (int i = 0; i < maps.Count; i++) Close(Values(expected[i]), maps[i], $"map {i}");
        Close(Values(fixture["embedding"]!), embedding, "embedding");
    }
}
