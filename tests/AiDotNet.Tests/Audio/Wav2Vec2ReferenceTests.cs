using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The shared wav2vec 2.0 network against transformers' Wav2Vec2Model (tools/reference-data/wav2vec2_reference.py), in
/// its BASE (group norm, post-norm) and LARGE-LV60 (layer norm, stable pre-norm) forms.
/// </summary>
public class Wav2Vec2ReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "wav2vec2_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
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

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task Forward_Features_AndOneSgdStep_MatchTheReference(int index)
    {
        await Task.Yield();
        var k = Fixture().GetProperty("cases")[index];
        bool layerNorm = k.GetProperty("layer_norm").GetBoolean();
        var configuration = new Wav2Vec2Configuration
        {
            ConvDims = new[] { 8, 8, 8 }, ConvKernels = new[] { 10, 3, 2 }, ConvStrides = new[] { 5, 2, 2 }, ConvBias = layerNorm,
            LayerNormFeatureEncoder = layerNorm, StableLayerNorm = layerNorm, HiddenSize = 16, Layers = 2, Heads = 2,
            IntermediateSize = 32, PositionalConvKernel = 8, PositionalConvGroups = 4,
        };
        var layers = new List<LayerBase<double>>();
        var net = new Wav2Vec2Network<double>(AiDotNetEngine.Current, layers, configuration);
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var audio = new Tensor<double>(new[] { 2, 400 }, new Vector<double>(Values(k.GetProperty("audio"))));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Wav2Vec2Network<double>.Output output;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> loss;
        using (var tape = new GradientTape<double>())
        {
            output = net.Forward(audio, training: false, new Random(0));
            loss = AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(output.Hidden, output.Hidden), new[] { 0, 1, 2 }, false);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        Assert.Equal(k.GetProperty("frames").GetInt32(), output.Hidden.Shape[1]);
        AssertClose(Values(k.GetProperty("extract_features")), output.Features, 1e-10, "extract_features");
        AssertClose(Values(k.GetProperty("last_hidden_state")), output.Hidden, 1e-9, "last_hidden_state");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-10);
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        var after = net.Forward(audio, training: false, new Random(0)).Hidden;
        double afterLoss = after.ToArray().Average(v => v * v);
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - afterLoss) < 1e-9,
            $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {afterLoss}");
    }
}
