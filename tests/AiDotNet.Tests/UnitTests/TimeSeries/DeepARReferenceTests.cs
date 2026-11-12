using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// The shared DeepAR network against a PyTorch transcription of the paper (tools/reference-data/deepar_reference.py):
/// the window loss under each likelihood, and the loss after one SGD step.
/// </summary>
public class DeepARReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "deepar_reference.json";
        string relative = Path.Combine("UnitTests", "TimeSeries", "ReferenceData", name);
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

    // The non-generic overload: Enum.Parse<TEnum> does not exist on net471, which this project also targets.
    private static TEnum ParseEnum<TEnum>(JsonElement e, string property) where TEnum : struct, Enum
        => (TEnum)Enum.Parse(typeof(TEnum),
            e.GetProperty(property).GetString() ?? throw new InvalidDataException($"Fixture field '{property}' is null."));

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public async Task WindowLoss_AndOneSgdStep_MatchTheReference(int index)
    {
        await Task.Yield();
        var f = Fixture();
        var c = f.GetProperty("config");
        int hidden = c.GetProperty("hidden").GetInt32(), depth = c.GetProperty("layers").GetInt32(), cov = c.GetProperty("covariates").GetInt32();
        int steps = c.GetProperty("steps").GetInt32(), conditioning = c.GetProperty("conditioning").GetInt32(), batch = c.GetProperty("batch").GetInt32();
        var k = f.GetProperty("cases")[index];
        var likelihood = ParseEnum<DeepARLikelihood>(k, "likelihood");
        var scaling = ParseEnum<DeepARScaleHandling>(k, "scaling");
        var layers = new List<LayerBase<double>>();
        var net = new DeepARNetwork<double>(AiDotNetEngine.Current, layers, new DeepARConfiguration(cov, hidden, depth, 0.0, likelihood, scaling));
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var w = Values(k.GetProperty("window"));
        var window = new double[batch, steps + 1];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t <= steps; t++) window[b, t] = w[b * (steps + 1) + t];
        var x = Values(k.GetProperty("covariates"));
        var covariates = new double[batch, steps, cov];
        for (int i = 0; i < x.Length; i++) covariates[i / (steps * cov), i / cov % steps, i % cov] = x[i];

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Tensor<double> loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            loss = net.Loss(window, covariates, conditioning, training: false, new Random(0));
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double expected = k.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(expected - loss[0]) < 1e-10, $"{likelihood}: {expected} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        var after = net.Loss(window, covariates, conditioning, training: false, new Random(0));
        double expectedAfter = k.GetProperty("loss_after_sgd").GetDouble();
        Assert.True(Math.Abs(expectedAfter - after[0]) < 1e-9, $"{likelihood} after SGD: {expectedAfter} vs {after[0]}");
    }
}
