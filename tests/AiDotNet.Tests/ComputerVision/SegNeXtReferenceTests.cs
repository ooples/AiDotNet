using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation;
using AiDotNet.ComputerVision.Segmentation.Semantic;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// SegNeXt against mmsegmentation's own MSCAN and LightHamHead source (tools/reference-data/segnext_reference.py):
/// the four stage outputs, the logits in evaluation (7 NMF steps) and training (BatchNorm batch statistics, 6 NMF
/// steps), the pixel cross-entropy after resizing the logits to the labels, and the loss after one SGD step. The NMF's
/// random bases come from the fixture so both sides factor from the same draw.
/// </summary>
public class SegNeXtReferenceTests
{
    private const int Size = 64;

    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "segnext_reference.json";
        string relative = Path.Combine("ComputerVision", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    private static Tensor<double> CrossEntropy(IEngine engine, Tensor<double> logits, Tensor<double> oneHot)
    {
        var up = SpatialResampling.Resize(engine, logits, Size, Size);
        var max = new Tensor<double>(new[] { up.Shape[0], 1, Size, Size });
        int classes = up.Shape[1], pixels = Size * Size;
        for (int b = 0; b < up.Shape[0]; b++)
            for (int p = 0; p < pixels; p++)
            {
                double m = double.NegativeInfinity;
                for (int c = 0; c < classes; c++) m = Math.Max(m, up[(b * classes + c) * pixels + p]);
                max[b * pixels + p] = m;
            }
        var centered = engine.TensorSubtract(up, engine.TensorBroadcastTo(max, up.Shape.ToArray()));
        var logSumExp = engine.TensorLog(engine.ReduceSum(engine.TensorExp(centered), new[] { 1 }, false));
        var picked = engine.ReduceSum(engine.TensorMultiply(centered, oneHot), new[] { 1 }, false);
        return engine.ReduceMean(engine.TensorSubtract(logSumExp, picked), new[] { 0, 1, 2 }, false);
    }

    [Fact(Timeout = 300000)]
    public async Task Network_MatchesTheOfficialCode_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Data.Value;
        var configuration = new SegNeXtConfiguration
        {
            EmbedDims = new[] { 8, 16, 32, 32 }, Depths = new[] { 1, 1, 2, 1 }, MlpRatios = new[] { 2, 2, 2, 2 },
            DropPathRate = 0.0, HeadChannels = 32, HamChannels = 32, Classes = 3, MatrixRank = 4, HeadDropout = 0.0,
        };
        var layers = new List<LayerBase<double>>();
        var net = new SegNeXtNetwork<double>(engine, layers, configuration);
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        foreach (var layer in layers) layer.SetTrainingMode(false);

        var images = new Tensor<double>(new[] { 2, 3, Size, Size }, new Vector<double>(Values(k.GetProperty("images"))));
        var labels = k.GetProperty("labels").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var oneHot = new Tensor<double>(new[] { 2, 3, Size, Size });
        for (int b = 0; b < 2; b++)
            for (int p = 0; p < Size * Size; p++) oneHot[(b * 3 + labels[b * Size * Size + p]) * Size * Size + p] = 1.0;
        var bases = new Tensor<double>(new[] { 2, 32, 4 }, new Vector<double>(Values(k.GetProperty("eval_bases"))));

        var features = net.Features(images, training: false, new Random(0));
        var shapes = k.GetProperty("feature_shapes").EnumerateArray().Select(s => s.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(shapes[i], features[i].Shape.ToArray());
            AssertClose(Values(k.GetProperty("features")[i]), features[i], 1e-9, $"stage {i + 1}");
        }

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> logits, loss;
        using (var tape = new GradientTape<double>())
        {
            logits = net.Forward(images, training: false, new Random(0), bases);
            loss = CrossEntropy(engine, logits, oneHot);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(k.GetProperty("logits")), logits, 1e-9, "eval logits");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9, $"loss {k.GetProperty("loss").GetDouble()} vs {loss[0]}");

        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = CrossEntropy(engine, net.Forward(images, training: false, new Random(0), bases), oneHot)[0];
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-9,
            $"loss after SGD {k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");

        foreach (var layer in layers) layer.SetTrainingMode(true);
        var trainBases = new Tensor<double>(new[] { 2, 32, 4 }, new Vector<double>(Values(k.GetProperty("train_bases"))));
        Tensor<double> trainLogits;
        using (new NoGradScope<double>())
            trainLogits = net.Forward(images, training: true, new Random(0), trainBases);
        AssertClose(Values(k.GetProperty("train_logits")), trainLogits, 1e-9, "train logits");
    }
}
