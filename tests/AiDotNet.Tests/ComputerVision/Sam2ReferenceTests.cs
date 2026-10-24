using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// The SAM 2 image network against Hugging Face's port of the official code (tools/reference-data/sam2_reference.py):
/// Hiera with windowed, global and query-pooling blocks, the FPN neck, the prompt encoder and the mask decoder, for point,
/// box, box+point and box+mask prompts with multimask output on and off, before and after one SGD step.
/// </summary>
public class Sam2ReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "sam2_reference.json";
        string relative = Path.Combine("ComputerVision", "ReferenceData", name);
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

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    internal static Sam2Configuration SmallConfiguration() => new()
    {
        ImageSize = 128, BlocksPerStage = new[] { 1, 1, 2, 1 }, EmbedDims = new[] { 16, 32, 64, 128 }, HeadsPerStage = new[] { 1, 2, 4, 8 },
        WindowSizes = new[] { 4, 4, 4, 2 }, GlobalAttentionBlocks = new[] { 3 }, Dim = 32, MaskInChannels = 8, DecoderHeads = 2,
        DecoderMlp = 64, IouHidden = 16,
    };

    [Fact(Timeout = 300000)]
    public async Task Network_MatchesTheOfficialCode_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var layers = new List<LayerBase<double>>();
        var net = new Sam2Network<double>(engine, layers, SmallConfiguration());
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var image = new Tensor<double>(new[] { 1, 3, 128, 128 }, new Vector<double>(Values(k.GetProperty("image"))));
        var lowMask = new Tensor<double>(new[] { 1, 1, 32, 32 }, new Vector<double>(Values(k.GetProperty("low_res_mask"))));
        var prompts = k.GetProperty("prompts").EnumerateArray().ToArray();

        List<(Tensor<double> Logits, Tensor<double> Iou, Tensor<double> Object)> Run(out (Tensor<double>, Tensor<double>, Tensor<double>) features)
        {
            features = net.EncodeImage(image);
            var (embedding, s0, s1) = features;
            var positional = net.DensePositional();
            var outputs = new List<(Tensor<double>, Tensor<double>, Tensor<double>)>();
            foreach (var p in prompts)
            {
                Tensor<double>? points = null, labels = null, boxes = null;
                if (p.TryGetProperty("points", out var pts))
                {
                    var list = pts.EnumerateArray().Select(Values).ToArray();
                    points = new Tensor<double>(new[] { 1, list.Length, 2 }, new Vector<double>(list.SelectMany(x => x).ToArray()));
                    labels = new Tensor<double>(new[] { 1, list.Length }, new Vector<double>(Values(p.GetProperty("labels"))));
                }
                if (p.TryGetProperty("box", out var box)) boxes = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(Values(box)));
                var masks = p.TryGetProperty("mask", out _) ? lowMask : null;
                var (sparse, dense) = net.EncodePrompts(points, labels, boxes, masks);
                foreach (bool multi in new[] { true, false })
                    outputs.Add(net.DecodeMasks(embedding, s0, s1, positional, sparse, dense, multi));
            }
            return outputs;
        }

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> loss;
        List<(Tensor<double> Logits, Tensor<double> Iou, Tensor<double> Object)> results;
        (Tensor<double> Embedding, Tensor<double> S0, Tensor<double> S1) features;
        using (var tape = new GradientTape<double>())
        {
            results = Run(out features);
            loss = results.Select(r => engine.TensorAdd(engine.ReduceMean(engine.TensorMultiply(r.Logits, r.Logits), Enumerable.Range(0, 4).ToArray(), false),
                engine.ReduceMean(engine.TensorMultiply(r.Iou, r.Iou), new[] { 0, 1 }, false))).Aggregate(engine.TensorAdd);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(k.GetProperty("embedding")), features.Embedding, 1e-9, "image embedding");
        AssertClose(Values(k.GetProperty("high_res0")), features.S0, 1e-9, "high-res 0");
        AssertClose(Values(k.GetProperty("high_res1")), features.S1, 1e-9, "high-res 1");
        var expected = k.GetProperty("results").EnumerateArray().ToArray();
        Assert.Equal(expected.Length, results.Count);
        for (int i = 0; i < results.Count; i++)
        {
            string what = $"{expected[i].GetProperty("prompt").GetString()} multimask={expected[i].GetProperty("multimask").GetBoolean()}";
            var shape = expected[i].GetProperty("logits_shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            Assert.Equal(shape, results[i].Logits.Shape.ToArray().Skip(1).ToArray());
            AssertClose(Values(expected[i].GetProperty("logits")), results[i].Logits, 1e-9, what + " logits");
            AssertClose(Values(expected[i].GetProperty("iou")), results[i].Iou, 1e-9, what + " iou");
            AssertClose(Values(expected[i].GetProperty("object_score")), results[i].Object, 1e-9, what + " object score");
        }
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.01 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Run(out _).Sum(r => r.Logits.ToArray().Average(v => v * v) + r.Iou.ToArray().Average(v => v * v));
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-9, $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");
    }
}
