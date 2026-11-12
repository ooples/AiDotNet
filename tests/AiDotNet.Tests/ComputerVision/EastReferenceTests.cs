using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// EAST against a PyTorch transcription of the paper's network with the VGG16 stem (tools/reference-data/east_reference.py):
/// the score and RBOX geometry maps, and the loss before and after one SGD step. The weights are generated from a hash
/// of each parameter's name, identically on both sides.
/// </summary>
public class EastReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "east_reference.json";
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

    /// <summary>The fixture's uniforms(): FNV-1a of the name, xorshift of the index.</summary>
    internal static double[] Uniforms(string name, int count)
    {
        uint h = 2166136261;
        foreach (byte c in Encoding.ASCII.GetBytes(name)) h = unchecked((h ^ c) * 16777619);
        var u = new double[count];
        for (int j = 0; j < count; j++)
        {
            uint x = h ^ unchecked((uint)j * 2654435761u);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            u[j] = x / 4294967296.0;
        }
        return u;
    }

    private static double[] Synthetic(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var u = Uniforms(name, count);
        if (name.EndsWith("running_var", StringComparison.Ordinal) || name.EndsWith("norm.weight", StringComparison.Ordinal))
            return u.Select(v => 0.5 + v).ToArray();
        if (name.EndsWith("running_mean", StringComparison.Ordinal) || name.EndsWith(".bias", StringComparison.Ordinal))
            return u.Select(v => 0.1 * (2 * v - 1)).ToArray();
        double bound = Math.Sqrt(6.0 / (count / shape[0]));
        return u.Select(v => bound * (2 * v - 1)).ToArray();
    }

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    [Fact(Timeout = 300000)]
    public async Task Network_MatchesTheTranscription_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var east = new EAST<double>(new TextDetectionOptions<double> { InputSize = new[] { 64, 96 }, Size = AiDotNet.Models.Options.ModelSize.Nano });
        east.LoadTorchWeights(Synthetic);
        east.SetTrainingMode(false);
        var images = new Tensor<double>(new[] { 2, 3, 64, 96 }, new Vector<double>(Uniforms("images", 2 * 3 * 64 * 96)));

        Tensor<double> Loss(List<Tensor<double>> maps) => engine.TensorAdd(
            engine.ReduceMean(engine.TensorMultiply(maps[0], maps[0]), new[] { 0, 1, 2, 3 }, false),
            engine.ReduceMean(engine.TensorMultiply(maps[1], maps[1]), new[] { 0, 1, 2, 3 }, false));

        var all = east.GetParameterStateChunks().Where(c => c.Role == AiDotNet.Models.Parameters.ParameterSlotRole.Trainable).Select(c => c.Tensor).ToList();
        Dictionary<Tensor<double>, Tensor<double>> grads;
        List<Tensor<double>> maps;
        Tensor<double> loss;
        using (var tape = new GradientTape<double>())
        {
            maps = east.Maps(images);
            loss = Loss(maps);
            grads = tape.ComputeGradients(loss, sources: all);
        }
        AssertClose(Values(k.GetProperty("score")), maps[0], 1e-9, "score");
        AssertClose(Values(k.GetProperty("geometry")), maps[1], 1e-9, "geometry");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-9 * k.GetProperty("loss").GetDouble(), $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        Assert.Equal(all.Count, grads.Count);
        foreach (var p in all)
        {
            var g = grads[p];
            for (int i = 0; i < p.Length; i++) p[i] -= 0.01 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Loss(east.Maps(images))[0];
        double expected = k.GetProperty("loss_after_sgd").GetDouble();
        Assert.True(Math.Abs(expected - after) < 1e-8 * Math.Max(1, expected), $"{expected} vs {after}");
    }
}
