using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.StyleEmotion;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// StyleTTS-ZS (Li et al. 2024) has no released code; its components reproduce the references its gaps follow: the
/// conformer stacks match wenet's ConformerEncoderLayer with ESPnet relative positions.
/// </summary>
public class StyleTTSZSPaperTests
{
    private static JObject Reference(string fileName)
    {
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

    /// <summary>A reader over a fixture's base64 safetensors that records which tensors were read.</summary>
    internal static (TorchReader Read, Func<IEnumerable<string>> Unused, Action Dispose) Weights(JObject fixture)
    {
        string path = Path.Combine(Path.GetTempPath(), $"styletts_zs_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        var weights = new WeightLoader().LoadWeights(path);
        File.Delete(path);
        var used = new HashSet<string>();
        TorchReader read = (name, shape) =>
        {
            Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), weights[name].Length);
            used.Add(name);
            return weights[name].ToVector().Select(v => (double)v).ToArray();
        };
        return (read, () => weights.Keys.Where(k => !used.Contains(k) && !k.EndsWith("num_batches_tracked", StringComparison.Ordinal)), () => { });
    }

    [Fact(Timeout = 120000)]
    public async Task Conformer_MatchesWenet()
    {
        await Task.Yield();
        var fixture = Reference("styletts_zs_conformer_reference.json");
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var layers = new List<LayerBase<double>>();
        int dim = (int)fixture["dim"]!;
        var stack = new StyleTtsZsConformer<double>(engine, layers, dim, (int)fixture["heads"]!, (int)fixture["feed_forward"]!,
            fixture["kernels"]!.Select(v => (int)v).ToArray(), 0.0, new Random(0));
        var input = Values(fixture["input"]!);
        var x = new Tensor<double>(new[] { input.Length / dim, dim }, new Vector<double>(input));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        stack.Forward(x, false, new Random(0));
        var (read, unused, _) = Weights(fixture);
        stack.Load(read, "encoders.");
        Assert.Empty(unused());
        Close(Values(fixture["output"]!), stack.Forward(x, false, new Random(0)), "conformer output");
    }
}
