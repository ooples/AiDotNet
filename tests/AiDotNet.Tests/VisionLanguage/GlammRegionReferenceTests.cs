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
using AiDotNet.Tests.ComputerVision;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// GLaMM's region encoder against a transcription of the official MLVLROIQueryModule (tools/reference-data/glamm_region_reference.py):
/// level upsampling, the five shuffle rounds, aligned RoIAlign on every level and the position embedding, for two images
/// with two and one boxes, before and after one SGD step. Weights and features are generated from their names on both sides.
/// </summary>
public class GlammRegionReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "glamm_region_reference.json";
        string relative = Path.Combine("VisionLanguage", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Synthetic(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var u = EastReferenceTests.Uniforms(name, count);
        if (name.EndsWith(".bias", StringComparison.Ordinal)) return u.Select(v => 0.1 * (2 * v - 1)).ToArray();
        if ((name.StartsWith("mlvl_fuse.fuse_convs.", StringComparison.Ordinal) && name.EndsWith(".1.weight", StringComparison.Ordinal))
            || name is "roi_align.pos_embedd.2.weight" or "roi_align.pos_embedd.5.weight")
            return u.Select(v => 0.5 + v).ToArray();
        double bound = Math.Sqrt(6.0 / (count / shape[0]));
        return u.Select(v => bound * (2 * v - 1)).ToArray();
    }

    [Fact(Timeout = 300000)]
    public async Task RegionEncoder_MatchesTheOfficialModule_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var layers = new List<LayerBase<double>>();
        var encoder = new GlammRegionEncoder<double>(engine, layers, new GlammRegionConfiguration { Embed = 64, Grid = 6, RegionWidth = 32, OutDims = 48 });
        encoder.LoadTorchWeights(Synthetic);
        var levels = Enumerable.Range(0, 4).Select(l => new Tensor<double>(new[] { 2, 36, 64 },
            new Vector<double>(EastReferenceTests.Uniforms($"features{l}", 2 * 36 * 64).Select(v => 2 * v - 1).ToArray()))).ToArray();
        var boxes = k.GetProperty("boxes").EnumerateArray()
            .Select(image => image.EnumerateArray().Select(b => b.EnumerateArray().Select(v => v.GetDouble()).ToArray()).ToArray()).ToArray();
        Tensor<double> Loss(Tensor<double> regions) => engine.ReduceMean(engine.TensorMultiply(regions, regions), new[] { 0, 1 }, false);

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Tensor<double> regions, loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            regions = encoder.Forward(levels, boxes);
            loss = Loss(regions);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        var expected = k.GetProperty("regions").EnumerateArray().SelectMany(r => r.EnumerateArray().Select(v => v.GetDouble())).ToArray();
        Assert.Equal(expected.Length, regions.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - regions[i]) < 1e-9 * Math.Max(1, Math.Abs(expected[i])), $"region[{i}]: {expected[i]} vs {regions[i]}");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9 * Math.Max(1, loss[0]), $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.01 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Loss(encoder.Forward(levels, boxes))[0];
        double reference = k.GetProperty("loss_after_sgd").GetDouble();
        Assert.True(Math.Abs(reference - after) < 1e-9 * Math.Max(1, reference), $"{reference} vs {after}");
    }
}
