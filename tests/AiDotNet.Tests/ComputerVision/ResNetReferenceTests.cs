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
/// The shared ResNet trunk, the image classifier and the detection backbone against torchvision's ResNet
/// (tools/reference-data/resnet_reference.py): a narrow basic-block and a narrow bottleneck network built from
/// torchvision's own classes, and the full networks' parameter counts.
/// </summary>
public class ResNetReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "resnet_reference.json";
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

    private static NeuralNetworkArchitecture<double> Architecture(int height, int width, int classes) => new(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputHeight: height, inputWidth: width, inputDepth: 3, outputSize: classes);

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task Classifier_MatchesTorchvision_InEvalAndTrainingMode_AndAfterOneSgdStep(int index)
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture().GetProperty("cases")[index];
        const int classes = 5;
        var variant = k.GetProperty("bottleneck").GetBoolean() ? ResNetVariant.ResNet50 : ResNetVariant.ResNet18;
        var configuration = new ResNetConfiguration(variant, classes, inputHeight: 40, inputWidth: 36)
            .WithShape(Ints(k.GetProperty("blocks")), Ints(k.GetProperty("widths")));
        var net = new ResNetNetwork<double>(Architecture(40, 36, classes), configuration);
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var images = new Tensor<double>(new[] { 2, 3, 40, 36 }, new Vector<double>(Values(k.GetProperty("images"))));
        var targets = Ints(k.GetProperty("targets"));
        var oneHot = new Tensor<double>(new[] { 2, classes });
        for (int b = 0; b < 2; b++) oneHot[b, targets[b]] = 1.0;

        net.SetTrainingMode(false);
        var features = net.Trunk.Forward(images);
        var expectedShapes = k.GetProperty("feature_shapes").EnumerateArray().Select(Ints).ToArray();
        for (int s = 0; s < 4; s++)
        {
            Assert.Equal(expectedShapes[s], features[s].Shape.ToArray());
            AssertClose(Values(k.GetProperty("features")[s]), features[s], 1e-9, $"C{s + 2}");
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

        net.SetTrainingMode(true);
        Tensor<double> trainProbabilities;
        using (new NoGradScope<double>())
            trainProbabilities = net.Forward(images);
        AssertClose(Softmax(Values(k.GetProperty("train_logits")), classes), trainProbabilities, 1e-9, "training-mode probabilities");
        AssertClose(Values(k.GetProperty("train_bn1_running_mean")), net.Trunk.StemNorm.GetRunningMean(), 1e-12, "bn1.running_mean");
        AssertClose(Values(k.GetProperty("train_bn1_running_var")), net.Trunk.StemNorm.GetRunningVariance(), 1e-12, "bn1.running_var");
        AssertClose(Values(k.GetProperty("train_last_running_var")), net.Trunk.Stages[3][^1].LastNorm.GetRunningVariance(), 1e-12,
            "last running_var");
    }

    [Theory(Timeout = 300000)]
    [InlineData(ResNetVariant.ResNet18, "resnet18")]
    [InlineData(ResNetVariant.ResNet34, "resnet34")]
    [InlineData(ResNetVariant.ResNet50, "resnet50")]
    [InlineData(ResNetVariant.ResNet101, "resnet101")]
    [InlineData(ResNetVariant.ResNet152, "resnet152")]
    public async Task ParameterCounts_MatchTorchvision(ResNetVariant variant, string name)
    {
        await Task.Yield();
        long expected = Fixture().GetProperty("parameter_counts").GetProperty(name).GetInt64();
        var classifier = new ResNetNetwork<float>(
            new NeuralNetworkArchitecture<float>(inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3,
                outputSize: 1000),
            new ResNetConfiguration(variant, 1000, inputHeight: 32, inputWidth: 32));
        // torchvision counts trainable tensors; this library's BatchNorm also counts its two running-statistic buffers.
        static long Trainable(IEnumerable<LayerBase<float>> layers) => layers.Sum(l => (long)l.GetTrainableParameters().Sum(t => t.Length));
        static long Buffers(IEnumerable<LayerBase<float>> layers) => layers.OfType<BatchNormalizationLayer<float>>().Sum(n => 2L * n.GetGamma().Length);
        var classifierLayers = classifier.Trunk.OwnedLayers.Append(classifier.Classifier!).ToList();
        Assert.Equal(expected, Trainable(classifierLayers));
        Assert.Equal(expected + Buffers(classifierLayers), classifier.GetParameterCount());
        int features = variant >= ResNetVariant.ResNet50 ? 2048 : 512;
        var backbone = new ResNet<float>(new ResNetBackboneOptions { Variant = variant });
        Assert.Equal(expected - (features * 1000 + 1000), Trainable(backbone.Trunk.OwnedLayers));
        Assert.Equal(Trainable(backbone.Trunk.OwnedLayers) + Buffers(backbone.Trunk.OwnedLayers), backbone.GetParameterCount());
    }

    [Fact(Timeout = 120000)]
    public async Task Backbone_RunsTheClassifierTrunk_AndRoundTripsItsRunningStatistics()
    {
        await Task.Yield();
        var random = new Random(7);
        var classifier = new ResNetNetwork<double>(Architecture(64, 64, 10), new ResNetConfiguration(ResNetVariant.ResNet18, 10, 64, 64));
        var backbone = new ResNet<double>(new ResNetBackboneOptions { Variant = ResNetVariant.ResNet18 });
        // Give the classifier non-trivial BatchNorm statistics, then copy its trunk into the backbone by name.
        var named = new Dictionary<string, double[]>();
        classifier.Trunk.LoadTorchWeights((name, shape) =>
        {
            int n = shape.Aggregate(1, (a, b) => a * b);
            var v = Enumerable.Range(0, n).Select(_ => name.EndsWith("running_var", StringComparison.Ordinal)
                ? 1 + random.NextDouble() : 0.1 * (random.NextDouble() - 0.5)).ToArray();
            named[name] = v;
            return v;
        });
        backbone.LoadTorchWeights((name, _) => named[name]);
        classifier.SetTrainingMode(false);
        backbone.SetTrainingMode(false);
        var image = new Tensor<double>(new[] { 1, 3, 64, 64 });
        for (int i = 0; i < image.Length; i++) image[i] = random.NextDouble() - 0.5;
        var expected = classifier.Trunk.Forward(image);
        var actual = backbone.ExtractFeatures(image);
        int[][] shapes = { new[] { 1, 64, 16, 16 }, new[] { 1, 128, 8, 8 }, new[] { 1, 256, 4, 4 }, new[] { 1, 512, 2, 2 } };
        for (int s = 0; s < 4; s++)
        {
            Assert.Equal(shapes[s], actual[s].Shape.ToArray());
            AssertClose(expected[s].ToArray(), actual[s], 1e-12, $"C{s + 2}");
        }

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true)) backbone.WriteParameters(writer);
        stream.Position = 0;
        var restored = new ResNet<double>(new ResNetBackboneOptions { Variant = ResNetVariant.ResNet18 });
        using (var reader = new BinaryReader(stream)) restored.ReadParameters(reader);
        restored.SetTrainingMode(false);
        var again = restored.ExtractFeatures(image);
        for (int s = 0; s < 4; s++) AssertClose(expected[s].ToArray(), again[s], 1e-12, $"restored C{s + 2}");
    }
}
