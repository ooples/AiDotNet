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
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// The Grounding DINO network against Hugging Face's port of the official code (tools/reference-data/grounding_dino_reference.py):
/// Swin and BERT backbones, feature enhancer, language-guided query selection and decoder, every decoder layer's logits
/// and boxes for two prompts (one padded), and the loss before and after one SGD step.
/// </summary>
public class GroundingDinoReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "grounding_dino_reference.json";
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

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? double.NaN : v.GetDouble()).ToArray();

    internal static GroundingDinoConfiguration SmallConfiguration() => new()
    {
        Backbone = new SwinConfiguration { EmbedDim = 16, Depths = new[] { 1, 1, 2, 1 }, Heads = new[] { 1, 2, 2, 4 }, WindowSize = 4, DropPathRate = 0 },
        Text = new BertConfiguration { VocabSize = 2100, Hidden = 32, Layers = 2, Heads = 2, Intermediate = 64, MaxPositions = 64 },
        Dim = 32, EncoderLayers = 2, DecoderLayers = 2, EncoderHeads = 4, DecoderHeads = 4, EncoderFfn = 64, DecoderFfn = 64,
        EncoderPoints = 2, DecoderPoints = 2, Queries = 10, MaxTextLength = 16,
    };

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (double.IsNaN(expected[i]))
            {
                Assert.True(actual[i] <= GroundingDinoNetwork<double>.Masked, $"{what}[{i}] should be masked: {actual[i]}");
                continue;
            }
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
        }
    }

    [Fact(Timeout = 600000)]
    public async Task Network_MatchesTheReference_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var layers = new List<LayerBase<double>>();
        var net = new GroundingDinoNetwork<double>(engine, layers, SmallConfiguration());
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var images = new Tensor<double>(new[] { 2, 3, 64, 64 }, new Vector<double>(Values(k.GetProperty("images"))));
        var idRows = k.GetProperty("ids").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        var maskRows = k.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32() != 0).ToArray()).ToArray();
        var ids = new int[2, idRows[0].Length];
        var mask = new bool[2, idRows[0].Length];
        for (int i = 0; i < 2; i++) for (int j = 0; j < idRows[0].Length; j++) { ids[i, j] = idRows[i][j]; mask[i, j] = maskRows[i][j]; }

        Tensor<double> Loss(GroundingDinoOutput<double> o)
        {
            Tensor<double>? total = null;
            int l = ids.GetLength(1);
            var valid = new Tensor<double>(new[] { 2, 10, 16 });
            int count = 0;
            for (int i = 0; i < 2; i++) for (int q = 0; q < 10; q++) for (int t = 0; t < l; t++) if (mask[i, t]) { valid[i, q, t] = 1; count++; }
            for (int layer = 0; layer < o.Logits.Length; layer++)
            {
                var boxes = engine.ReduceMean(engine.TensorMultiply(o.Boxes[layer], o.Boxes[layer]), new[] { 0, 1, 2 }, false);
                var masked = engine.TensorMultiply(o.Logits[layer], valid);
                var logits = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(masked, masked), new[] { 0, 1, 2 }, false), 1.0 / count);
                var term = engine.TensorAdd(boxes, logits);
                total = total is null ? term : engine.TensorAdd(total, term);
            }
            return total!;
        }

        // Stage by stage first, so a mismatch names the first component that differs.
        net.Trace = new Dictionary<string, Tensor<double>>();
        using (new NoGradScope<double>())
            net.Forward(images, ids, mask);
        var trace = k.GetProperty("trace");
        var stages = new[] { "bert", "text_projected", "swin0", "swin1", "swin2", "projected0", "projected1", "projected2", "projected3",
            "fusion0_vision", "fusion0_text", "enhancer0", "deformable0", "encoder_vision", "encoder_text", "init_reference", "intermediate" };
        var errors = stages.Select(stage =>
        {
            var expected = Values(trace.GetProperty(stage));
            var actual = net.Trace[stage];
            return (stage, error: expected.Length != actual.Length ? double.PositiveInfinity
                : expected.Select((v, i) => Math.Abs(v - actual[i])).Max());
        }).ToArray();
        Assert.True(errors.All(e => e.error < 1e-8), string.Join("; ", errors.Select(e => $"{e.stage} {e.error:G3}")));
        net.Trace = null;

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        GroundingDinoOutput<double> output;
        Tensor<double> loss;
        using (var tape = new GradientTape<double>())
        {
            output = net.Forward(images, ids, mask);
            loss = Loss(output);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        var expectedLogits = k.GetProperty("logits").EnumerateArray().ToArray();
        var expectedBoxes = k.GetProperty("boxes").EnumerateArray().ToArray();
        for (int layer = 0; layer < 2; layer++)
        {
            AssertClose(Values(expectedBoxes[layer]), output.Boxes[layer], 1e-8, $"boxes {layer}");
            AssertClose(Values(expectedLogits[layer]), output.Logits[layer], 1e-8, $"logits {layer}");
        }
        AssertClose(Values(k.GetProperty("final_boxes")), output.Boxes[^1], 1e-8, "final boxes");
        AssertClose(Values(k.GetProperty("final_logits")), output.Logits[^1], 1e-8, "final logits");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-8, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.001 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Loss(net.Forward(images, ids, mask))[0];
        double reference = k.GetProperty("loss_after_sgd").GetDouble();
        Assert.True(Math.Abs(reference - after) < 1e-7 * Math.Max(1, reference), $"{reference} vs {after}");
    }
}
