using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// The shared EfficientNet trunk, the image classifier and the detection backbone against timm's tf_efficientnet, the
/// port of the official TensorFlow code (tools/reference-data/efficientnet_reference.py): a narrow network from timm's
/// own generator, and the full B0–B7 parameter counts.
/// </summary>
public class EfficientNetReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "efficientnet_reference.json";
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

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

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

    private static long Trainable(IEnumerable<LayerBase<float>> layers) => layers.Sum(l => (long)l.GetTrainableParameters().Sum(t => t.Length));

    private static long Buffers(IEnumerable<LayerBase<float>> layers) => layers.OfType<BatchNormalizationLayer<float>>().Sum(n => 2L * n.GetGamma().Length);

    [Fact(Timeout = 120000)]
    public async Task Classifier_MatchesTimm_InEvalAndTrainingMode_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        const int classes = 5;
        var configuration = new EfficientNetConfiguration(EfficientNetVariant.Custom, classes, customInputHeight: 34,
            customWidthMultiplier: 0.25, customDepthMultiplier: 0.3) { DropConnectRate = 0.0 };
        var architecture = new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: 34, inputWidth: 30, inputDepth: 3,
            outputSize: classes);
        var net = new EfficientNetNetwork<double>(architecture, configuration);
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var images = new Tensor<double>(new[] { 2, 3, 34, 30 }, new Vector<double>(Values(k.GetProperty("images"))));
        var targets = Ints(k.GetProperty("targets"));
        var oneHot = new Tensor<double>(new[] { 2, classes });
        for (int b = 0; b < 2; b++) oneHot[b, targets[b]] = 1.0;

        net.SetTrainingMode(false);
        var features = net.Trunk.Forward(images, training: false, new Random(0));
        var expectedShapes = k.GetProperty("feature_shapes").EnumerateArray().Select(Ints).ToArray();
        Assert.Equal(expectedShapes.Length, features.Count);
        for (int s = 0; s < features.Count; s++)
        {
            Assert.Equal(expectedShapes[s], features[s].Shape.ToArray());
            AssertClose(Values(k.GetProperty("features")[s]), features[s], 1e-9, $"stage {s}");
        }

        var layers = net.Trunk.OwnedLayers.Append(net.Classifier!).ToList();
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> probabilities, loss;
        using (var tape = new GradientTape<double>())
        {
            probabilities = net.Forward(images);
            var picked = engine.TensorMultiply(engine.TensorLog(probabilities), oneHot);
            loss = engine.TensorMultiplyScalar(engine.ReduceSum(picked, new[] { 0, 1 }, false), -0.5);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Softmax(Values(k.GetProperty("logits")), classes), probabilities, 1e-9, "probabilities");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        var after = net.Forward(images);
        double afterLoss = -Enumerable.Range(0, 2).Average(b => Math.Log(after[b, targets[b]]));
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - afterLoss) < 1e-9,
            $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {afterLoss}");

        // Training mode: batch statistics, the paper's BatchNorm momentum 0.99, and (with the configured rates at zero)
        // no dropout or stochastic depth, as in the fixture.
        net.SetTrainingMode(true);
        Tensor<double> trainFeatures;
        using (new NoGradScope<double>())
            trainFeatures = net.Trunk.Forward(images, training: true, new Random(0))[^1];
        var pooled = engine.ReduceMean(trainFeatures, new[] { 2, 3 }, false);
        Tensor<double> trainProbabilities;
        using (new NoGradScope<double>())
            trainProbabilities = net.Classifier!.Forward(pooled);
        AssertClose(Softmax(Values(k.GetProperty("train_logits")), classes), trainProbabilities, 1e-9, "training-mode probabilities");
        AssertClose(Values(k.GetProperty("train_bn1_running_mean")), net.Trunk.StemNorm.GetRunningMean(), 1e-12, "bn1.running_mean");
        AssertClose(Values(k.GetProperty("train_bn1_running_var")), net.Trunk.StemNorm.GetRunningVariance(), 1e-12, "bn1.running_var");
        AssertClose(Values(k.GetProperty("train_head_running_var")), net.Trunk.HeadNorm!.GetRunningVariance(), 1e-12, "bn2.running_var");
    }

    [Theory(Timeout = 300000)]
    [InlineData(EfficientNetVariant.B0, "b0")]
    [InlineData(EfficientNetVariant.B1, "b1")]
    [InlineData(EfficientNetVariant.B2, "b2")]
    [InlineData(EfficientNetVariant.B3, "b3")]
    [InlineData(EfficientNetVariant.B4, "b4")]
    [InlineData(EfficientNetVariant.B5, "b5")]
    [InlineData(EfficientNetVariant.B6, "b6")]
    [InlineData(EfficientNetVariant.B7, "b7")]
    public async Task ParameterCounts_MatchTimm(EfficientNetVariant variant, string name)
    {
        await Task.Yield();
        long expected = Fixture().GetProperty("parameter_counts").GetProperty(name).GetInt64();
        var configuration = new EfficientNetConfiguration(variant, 1000);
        var classifier = new EfficientNetNetwork<float>(new NeuralNetworkArchitecture<float>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 1000), configuration);
        var layers = classifier.Trunk.OwnedLayers.Append(classifier.Classifier!).ToList();
        Assert.Equal(expected, Trainable(layers));
        Assert.Equal(expected + Buffers(layers), classifier.GetParameterCount());
    }

    [Fact(Timeout = 120000)]
    public async Task Backbone_RunsTheClassifierTrunk_AtStrides4To32_AndRoundTripsItsState()
    {
        await Task.Yield();
        var random = new Random(7);
        var backbone = new EfficientNet<double>(new EfficientNetBackboneOptions { Variant = EfficientNetVariant.B0 });
        Assert.Equal(new[] { 24, 40, 112, 320 }, backbone.OutputChannels.ToArray());
        backbone.LoadTorchWeights((name, shape) =>
        {
            int n = shape.Aggregate(1, (a, b) => a * b);
            return Enumerable.Range(0, n).Select(_ => name.EndsWith("running_var", StringComparison.Ordinal)
                ? 1 + random.NextDouble() : 0.1 * (random.NextDouble() - 0.5)).ToArray();
        });
        backbone.SetTrainingMode(false);
        var image = new Tensor<double>(new[] { 1, 3, 64, 64 });
        for (int i = 0; i < image.Length; i++) image[i] = random.NextDouble() - 0.5;
        var features = backbone.ExtractFeatures(image);
        int[][] shapes = { new[] { 1, 24, 16, 16 }, new[] { 1, 40, 8, 8 }, new[] { 1, 112, 4, 4 }, new[] { 1, 320, 2, 2 } };
        for (int s = 0; s < 4; s++) Assert.Equal(shapes[s], features[s].Shape.ToArray());
        var stages = backbone.Trunk.Forward(image, training: false, new Random(0));
        AssertClose(stages[4].ToArray(), features[2], 1e-15, "C4 is the end of stage 5");

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true)) backbone.WriteParameters(writer);
        stream.Position = 0;
        var restored = new EfficientNet<double>(new EfficientNetBackboneOptions { Variant = EfficientNetVariant.B0 });
        using (var reader = new BinaryReader(stream)) restored.ReadParameters(reader);
        restored.SetTrainingMode(false);
        var again = restored.ExtractFeatures(image);
        for (int s = 0; s < 4; s++) AssertClose(features[s].ToArray(), again[s], 1e-12, $"restored C{s + 2}");
    }
}
