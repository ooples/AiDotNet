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
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// The shared Chronos-T5 core against the official chronos code and Hugging Face's T5 (tools/reference-data/
/// chronos_reference.py): the MeanScaleUniformBins tokenizer, the batch cross-entropy with masked context and ignored
/// labels, the loss after one SGD step, and the token path of top-k = 1 sampling.
/// </summary>
public class ChronosReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "chronos_reference.json";
        string relative = Path.Combine("UnitTests", "TimeSeries", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) =>
        e.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? double.NaN : v.GetDouble()).ToArray();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static bool[] Bools(JsonElement e) => e.EnumerateArray().Select(v => v.GetBoolean()).ToArray();

    private static ChronosConfiguration Configuration(int topK = 50, int samples = 20) =>
        new(ContextLength: 12, PredictionLength: 4, NumTokens: 64, LowLimit: -15, HighLimit: 15, ModelDim: 16,
            FeedForwardDim: 32, KeyValueDim: 8, Heads: 2, Layers: 2, Dropout: 0, RelativeBuckets: 8, RelativeMaxDistance: 16,
            NumSamples: samples, Temperature: 1, TopK: topK, TopP: 1);

    private static (ChronosT5<double> core, List<LayerBase<double>> layers) Core(JsonElement state, ChronosConfiguration configuration)
    {
        var layers = new List<LayerBase<double>>();
        var core = new ChronosT5<double>(AiDotNetEngine.Current, layers, configuration);
        core.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        return (core, layers);
    }

    [Fact]
    public void Tokenizer_MatchesMeanScaleUniformBins()
    {
        var f = Fixture();
        var released = new ChronosTokenizer(4096, -15, 15, 512);
        var expected = Values(f.GetProperty("released_centers"));
        Assert.Equal(expected.Length, released.Centers.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal((float)expected[i], released.Centers[i]);

        var tokenizer = new ChronosTokenizer(64, -15, 15, 12);
        var centers = Values(f.GetProperty("centers"));
        for (int i = 0; i < centers.Length; i++) Assert.Equal((float)centers[i], tokenizer.Centers[i]);
        var boundaries = Values(f.GetProperty("boundaries"));
        for (int i = 0; i < boundaries.Length; i++) Assert.Equal((float)boundaries[i], tokenizer.Boundaries[i + 1]);

        var series = Values(f.GetProperty("series"));
        var (ids, mask, scale) = tokenizer.Context(series.Take(16).ToArray());
        Assert.Equal(Ints(f.GetProperty("context_ids")), ids);
        Assert.Equal(Bools(f.GetProperty("context_mask")), mask);
        Assert.Equal((float)f.GetProperty("scale").GetDouble(), scale);
        var (labelIds, labelMask) = tokenizer.Label(series.Skip(16).Take(4).ToArray(), scale);
        Assert.Equal(Ints(f.GetProperty("label_ids")), labelIds);
        Assert.Equal(Bools(f.GetProperty("label_mask")), labelMask);

        // The reference left-pads the short context to 12 with masked PAD tokens; unpadded, the rest is the same.
        var (shortIds, shortMask, shortScale) = tokenizer.Context(Values(f.GetProperty("short")));
        Assert.Equal(Ints(f.GetProperty("short_ids")).Skip(6), shortIds);
        Assert.Equal(Bools(f.GetProperty("short_mask")).Skip(6), shortMask);
        Assert.Equal((float)f.GetProperty("short_scale").GetDouble(), shortScale);
        Assert.Equal(Ints(f.GetProperty("short_label_ids")), tokenizer.Label(Values(f.GetProperty("short_future")), shortScale).Ids);

        var values = Values(f.GetProperty("values"));
        for (int token = 0; token < values.Length; token++) Assert.Equal((float)values[token], (float)tokenizer.Value(token, scale));
    }

    [Fact(Timeout = 120000)]
    public async Task BatchLoss_AndOneSgdStep_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var (core, layers) = Core(f.GetProperty("state"), Configuration());
        var series = Values(f.GetProperty("series"));
        var windows = new List<(double[] Context, double[] Future)>
        {
            (series.Take(16).ToArray(), series.Skip(16).Take(4).ToArray()),
            (Values(f.GetProperty("short")), Values(f.GetProperty("short_future"))),
        };
        var random = new Random(0);
        IReadOnlyList<Tensor<double>> parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Tensor<double> loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            loss = core.BatchLoss(windows, training: false, random);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        Assert.True(Math.Abs(f.GetProperty("loss").GetDouble() - loss[0]) < 1e-12, $"{f.GetProperty("loss").GetDouble()} vs {loss[0]}");

        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        var after = core.BatchLoss(windows, training: false, random);
        Assert.True(Math.Abs(f.GetProperty("loss_after_sgd").GetDouble() - after[0]) < 1e-12,
            $"{f.GetProperty("loss_after_sgd").GetDouble()} vs {after[0]}");
    }

    [Fact(Timeout = 120000)]
    public async Task TopOneSampling_FollowsTheReferencePath()
    {
        await Task.Yield();
        var f = Fixture();
        var (core, _) = Core(f.GetProperty("generator_state"), Configuration(topK: 1, samples: 2));
        var context = Values(f.GetProperty("series")).Take(16).ToArray();
        var scale = core.Tokenizer.Context(context).Scale;
        var expected = Ints(f.GetProperty("generated")).Select(t => core.Tokenizer.Value(t, scale)).ToArray();
        var paths = core.SamplePaths(context, 4, new Random(3));
        Assert.Equal(2, paths.Length);
        foreach (var path in paths) Assert.Equal(expected, path);
    }

    [Fact]
    public void MedianAndQuantile_FollowTorch()
    {
        Assert.Equal(2.0, ChronosT5<double>.Median(new[] { 4.0, 1.0, 2.0, 3.0 }));
        Assert.Equal(3.0, ChronosT5<double>.Median(new[] { 5.0, 1.0, 3.0 }));
        Assert.Equal(1.75, ChronosT5<double>.Quantile(new[] { 1.0, 2.0, 3.0, 4.0 }, 0.25), 12);
    }
}
