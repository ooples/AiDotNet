using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Mimi's training forward against the release's MimiModel in training mode (tools/reference-data/mimi_training_reference.py):
/// two steps' reconstructions, semantic quantizer outputs, commitment penalties and codebook moving averages.
/// </summary>
public class MimiTrainingReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative))).RootElement;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static void Close(double[] expected, IReadOnlyList<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Count);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]:R} vs {actual[i]:R}");
    }

    [Fact(Timeout = 600000)]
    public async Task TrainingForward_MatchesTheRelease()
    {
        await Task.Yield();
        var reference = Load("mimi_training_reference.json");
        var shapes = reference.GetProperty("shapes").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray(), StringComparer.Ordinal);
        var configuration = new MimiConfiguration
        {
            SampleRate = 1600, Dimension = 16, Filters = 4, Ratios = new[] { 4, 2 }, TransformerLayers = 1, Heads = 2, FeedForward = 32,
            Context = 6, Codebooks = 4, CodebookSize = 8, CodebookDim = 8,
        };
        var mimi = new Mimi<double>(new CpuEngine(), configuration, new List<LayerBase<double>>());
        var used = mimi.Load(name => shapes.TryGetValue(name, out var shape) ? MimiReferenceTests.ClosedForm(name, shape) : null);
        Assert.Empty(shapes.Keys.Where(n => !used.Contains(n)));
        var input = Doubles(reference.GetProperty("input"));
        var audio = new Tensor<double>(new[] { 1, 1, input.Length }, new Vector<double>(input));
        var random = new Random(0);
        int step = 0;
        foreach (var expected in reference.GetProperty("steps").EnumerateArray())
        {
            var (output, semantic, penalty) = mimi.TrainingForward(audio, 3, 0.0, random);
            Close(Doubles(expected.GetProperty("audio")), output.ToArray(), $"step {step} audio");
            Close(Doubles(expected.GetProperty("semantic")), semantic.ToArray(), $"step {step} semantic");
            Close(new[] { expected.GetProperty("penalty").GetDouble() }, new[] { penalty[0] }, $"step {step} penalty");
            var books = expected.GetProperty("codebooks").EnumerateArray().ToArray();
            var actual = mimi.AllCodebooks.ToArray();
            for (int q = 0; q < books.Length; q++)
            {
                Close(Doubles(books[q].GetProperty("sum")), actual[q].Sum, $"step {step} codebook {q} sum");
                Close(Doubles(books[q].GetProperty("usage")), actual[q].Usage, $"step {step} codebook {q} usage");
            }
            step++;
        }
    }
}
