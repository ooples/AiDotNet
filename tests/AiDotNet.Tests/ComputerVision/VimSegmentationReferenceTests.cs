using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Mamba;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// The Vision Mamba segmenter against the official VisionMambaSeg with mmsegmentation's UPerHead and FCNHead
/// (tools/reference-data/vim_segmentation_reference.py), as the Tiny configuration builds it and with the Small
/// configuration's 2D rotary embedding.
/// </summary>
public class VimSegmentationReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "vim_segmentation_reference.json";
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

    /// <summary>Mean cross-entropy of NCHW logits against a one-hot NCHW target.</summary>
    private static Tensor<double> CrossEntropy(IEngine engine, Tensor<double> logits, Tensor<double> oneHot)
    {
        var logSumExp = engine.TensorLog(engine.ReduceSum(engine.TensorExp(logits), new[] { 1 }, false));
        var picked = engine.ReduceSum(engine.TensorMultiply(logits, oneHot), new[] { 1 }, false);
        return engine.ReduceMean(engine.TensorSubtract(logSumExp, picked), new[] { 0, 1, 2 }, false);
    }

    [Theory(Timeout = 300000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task Segmenter_MatchesTheOfficialCode_AndAfterOneSgdStep(int index)
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture().GetProperty("cases")[index];
        bool rope = k.GetProperty("rope").GetBoolean();
        var configuration = new VimSegmentationConfiguration
        {
            Backbone = new VimConfiguration
            {
                ImageHeight = 64, ImageWidth = 64, PatchSize = 16, Stride = 16, Dim = 16, Depth = 4, StateDim = 4,
                DropPathRate = 0.0, ClassToken = false, Rope = rope, RopeResidual = rope,
            },
            OutIndices = new[] { 0, 1, 2, 3 }, Classes = 3, HeadChannels = 16, AuxiliaryChannels = 16,
        };
        var layers = new List<LayerBase<double>>();
        var net = new VimSegmentationNetwork<double>(engine, layers, configuration);
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        foreach (var layer in layers) layer.SetTrainingMode(false);
        var images = new Tensor<double>(new[] { 2, 3, 64, 64 }, new Vector<double>(Values(k.GetProperty("images"))));
        var labels = k.GetProperty("labels").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var oneHot = new Tensor<double>(new[] { 2, 3, 64, 64 });
        for (int b = 0; b < 2; b++)
            for (int p = 0; p < 64 * 64; p++) oneHot[(b * 3 + labels[b * 4096 + p]) * 4096 + p] = 1.0;

        var features = net.Features(images, training: false, new Random(0));
        var shapes = k.GetProperty("feature_shapes").EnumerateArray().Select(s => s.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(shapes[i], features[i].Shape.ToArray());
            AssertClose(Values(k.GetProperty("features")[i]), features[i], 1e-9, $"feature {i}");
        }

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> logits, loss;
        Tensor<double>? auxiliary;
        using (var tape = new GradientTape<double>())
        {
            (logits, auxiliary) = net.Forward(images, training: false, new Random(0), withAuxiliary: true);
            loss = engine.TensorAdd(CrossEntropy(engine, logits, oneHot),
                engine.TensorMultiplyScalar(CrossEntropy(engine, auxiliary!, oneHot), 0.4));
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(k.GetProperty("decode_logits")), logits, 1e-9, "decode logits");
        AssertClose(Values(k.GetProperty("auxiliary_logits")), auxiliary!, 1e-9, "auxiliary logits");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
        {
            var (l2, a2) = net.Forward(images, training: false, new Random(0), withAuxiliary: true);
            after = CrossEntropy(engine, l2, oneHot)[0] + 0.4 * CrossEntropy(engine, a2!, oneHot)[0];
        }
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-9,
            $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");
    }
}
