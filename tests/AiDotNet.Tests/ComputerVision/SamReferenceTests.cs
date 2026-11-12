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
/// The shared Segment Anything network against the official segment_anything package
/// (tools/reference-data/sam_reference.py): image encoder, prompt encoder and mask decoder for point, box, box+point and
/// box+mask prompts, single and multimask, with postprocessing and one SGD step.
/// </summary>
public class SamReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "sam_reference.json";
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

    [Fact(Timeout = 300000)]
    public async Task Network_MatchesTheOfficialCode_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var configuration = new SamConfiguration
        {
            ImageSize = 96, PatchSize = 16, EncoderDim = 32, EncoderDepth = 3, EncoderHeads = 2, WindowSize = 4,
            GlobalAttention = new[] { 1 }, OutChannels = 16, MaskInChannels = 8, DecoderDepth = 2, DecoderHeads = 2,
            DecoderMlp = 32, IouHidden = 16,
        };
        var layers = new List<LayerBase<double>>();
        var net = new SamNetwork<double>(engine, layers, configuration);
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var image = new Tensor<double>(new[] { 1, 3, 80, 72 }, new Vector<double>(Values(k.GetProperty("image"))));
        var lowMask = new Tensor<double>(new[] { 1, 1, 24, 24 }, new Vector<double>(Values(k.GetProperty("low_res_mask"))));
        var prompts = k.GetProperty("prompts").EnumerateArray().ToArray();

        List<(Tensor<double> Logits, Tensor<double> Iou, Tensor<double> Masks)> Run(out Tensor<double> embedding, out Tensor<double> positional)
        {
            embedding = net.EncodeImage(net.Preprocess(image));
            positional = net.DensePositional();
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
                {
                    var (logits, iou) = net.DecodeMasks(embedding, positional, sparse, dense, multi);
                    outputs.Add((logits, iou, net.Postprocess(logits, 80, 72, 80, 72)));
                }
            }
            return outputs;
        }

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> loss;
        List<(Tensor<double> Logits, Tensor<double> Iou, Tensor<double> Masks)> results;
        Tensor<double> embedding, positional;
        using (var tape = new GradientTape<double>())
        {
            results = Run(out embedding, out positional);
            loss = results.Select(r => engine.TensorAdd(engine.ReduceMean(engine.TensorMultiply(r.Logits, r.Logits), Enumerable.Range(0, 4).ToArray(), false),
                engine.ReduceMean(engine.TensorMultiply(r.Iou, r.Iou), new[] { 0, 1 }, false))).Aggregate(engine.TensorAdd);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(k.GetProperty("embedding")), embedding, 1e-9, "image embedding");
        AssertClose(Values(k.GetProperty("dense_pe")), positional, 1e-9, "dense positional encoding");
        var expected = k.GetProperty("results").EnumerateArray().ToArray();
        Assert.Equal(expected.Length, results.Count);
        for (int i = 0; i < results.Count; i++)
        {
            string what = $"{expected[i].GetProperty("prompt").GetString()} multimask={expected[i].GetProperty("multimask").GetBoolean()}";
            Assert.Equal(expected[i].GetProperty("logits_shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(), results[i].Logits.Shape.ToArray());
            AssertClose(Values(expected[i].GetProperty("logits")), results[i].Logits, 1e-9, what + " logits");
            AssertClose(Values(expected[i].GetProperty("iou")), results[i].Iou, 1e-9, what + " iou");
            AssertClose(Values(expected[i].GetProperty("masks")), results[i].Masks, 1e-9, what + " masks");
        }
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Run(out _, out _).Sum(r => r.Logits.ToArray().Average(v => v * v) + r.Iou.ToArray().Average(v => v * v));
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-9, $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");
    }
}
