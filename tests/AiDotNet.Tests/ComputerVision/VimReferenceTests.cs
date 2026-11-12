using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// The shared Vision Mamba network against the official code (hustvl/Vim, tools/reference-data/vim_reference.py), with
/// non-overlapping and overlapping patches.
/// </summary>
public class VimReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "vim_reference.json";
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

    private static double[] Softmax(double[] logits, int classes)
    {
        var p = new double[logits.Length];
        for (int row = 0; row < logits.Length / classes; row++)
        {
            double max = Enumerable.Range(0, classes).Max(j => logits[row * classes + j]);
            double sum = Enumerable.Range(0, classes).Sum(j => Math.Exp(logits[row * classes + j] - max));
            for (int j = 0; j < classes; j++) p[row * classes + j] = Math.Exp(logits[row * classes + j] - max) / sum;
        }
        return p;
    }

    [Theory(Timeout = 300000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task Network_MatchesTheOfficialCode_AndAfterOneSgdStep(int index)
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture().GetProperty("cases")[index];
        const int classes = 5, dim = 16;
        var configuration = new VimConfiguration
        {
            ImageHeight = 16, ImageWidth = 16, PatchSize = k.GetProperty("patch").GetInt32(), Stride = k.GetProperty("stride").GetInt32(),
            Dim = dim, Depth = 2, StateDim = 4, DropPathRate = 0.0,
        };
        Assert.Equal(k.GetProperty("tokens").GetInt32(), configuration.Tokens);
        var layers = new List<LayerBase<double>>();
        var net = new VimNetwork<double>(engine, layers, configuration);
        var head = new DenseLayer<double>(classes, (IActivationFunction<double>)new SoftmaxActivation<double>());
        var state = k.GetProperty("state");
        double[] Read(string name, int[] shape)
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        }
        net.LoadTorchWeights(Read);
        TorchParameters.Linear(engine, head, dim, classes, Read("head.weight", new[] { classes, dim }), Read("head.bias", new[] { classes }));

        var images = new Tensor<double>(new[] { 2, 3, 16, 16 }, new Vector<double>(Values(k.GetProperty("images"))));
        var targets = k.GetProperty("targets").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var oneHot = new Tensor<double>(new[] { 2, classes });
        for (int b = 0; b < 2; b++) oneHot[b, targets[b]] = 1.0;

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Append(head).ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> features, probabilities, loss;
        using (var tape = new GradientTape<double>())
        {
            features = net.ClassFeatures(net.Forward(images, training: false, new Random(0)));
            probabilities = head.Forward(features);
            var picked = engine.TensorMultiply(engine.TensorLog(probabilities), oneHot);
            loss = engine.TensorMultiplyScalar(engine.ReduceSum(picked, new[] { 0, 1 }, false), -0.5);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(k.GetProperty("features")), features, 1e-9, "class-token features");
        AssertClose(Softmax(Values(k.GetProperty("logits")), classes), probabilities, 1e-9, "probabilities");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        Tensor<double> after;
        using (new NoGradScope<double>())
            after = head.Forward(net.ClassFeatures(net.Forward(images, training: false, new Random(0))));
        double afterLoss = -Enumerable.Range(0, 2).Average(b => Math.Log(after[b, targets[b]]));
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - afterLoss) < 1e-9,
            $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {afterLoss}");
    }
}
