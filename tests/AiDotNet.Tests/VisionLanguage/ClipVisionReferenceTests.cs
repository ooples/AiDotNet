using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// CLIP's vision transformer (GLaMM's global image encoder, through LLaVA) against Hugging Face's CLIPVisionModel
/// (tools/reference-data/clip_vision_reference.py): every hidden state, before and after one SGD step.
/// </summary>
public class ClipVisionReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "clip_vision_reference.json";
        string relative = Path.Combine("NeuralNetworks", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 300000)]
    public async Task VisionTower_MatchesTheReference_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var layers = new List<LayerBase<double>>();
        var net = new ClipVisionNetwork<double>(engine, layers, new ClipVisionConfiguration
        {
            ImageSize = 28, PatchSize = 7, Hidden = 32, Layers = 4, Heads = 4, Intermediate = 64,
        });
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        }, prefix: "");
        var images = new Tensor<double>(new[] { 2, 3, 28, 28 }, new Vector<double>(Values(k.GetProperty("images"))));
        Tensor<double> Loss(List<Tensor<double>> states) => states
            .Select(s => engine.ReduceMean(engine.TensorMultiply(s, s), new[] { 0, 1, 2 }, false)).Aggregate(engine.TensorAdd);

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        List<Tensor<double>> states;
        Tensor<double> loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            states = net.Forward(images);
            loss = Loss(states);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        var expected = k.GetProperty("states").EnumerateArray().Select(Values).ToArray();
        Assert.Equal(expected.Length, states.Count);
        for (int s = 0; s < expected.Length; s++)
        {
            Assert.Equal(expected[s].Length, states[s].Length);
            for (int i = 0; i < expected[s].Length; i++)
                Assert.True(Math.Abs(expected[s][i] - states[s][i]) < 1e-10, $"state {s}[{i}]: {expected[s][i]} vs {states[s][i]}");
        }
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-10, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.01 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Loss(net.Forward(images))[0];
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-10, $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");
    }
}
