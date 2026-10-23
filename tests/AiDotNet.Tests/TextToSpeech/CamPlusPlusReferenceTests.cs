using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Audio.Speaker;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CAM++, CosyVoice's x-vector extractor, against modelscope/3D-Speaker's <c>CAMPPlus</c>
/// (<c>tools/reference-data/campplus_reference.py</c>): eval-mode embeddings, the training-mode loss with batch
/// statistics, and the eval-mode embeddings after one SGD step (weights and running statistics both moved).
/// </summary>
public class CamPlusPlusReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "campplus_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static (CamPlusPlusEncoder<double> Encoder, List<LayerBase<double>> Layers, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        var configuration = new CamPlusPlusConfiguration(FeatureDim: (int)c["feat_dim"]!, EmbeddingSize: (int)c["embedding_size"]!,
            GrowthRate: (int)c["growth_rate"]!, BottleneckMultiplier: (int)c["bn_size"]!, InitialChannels: (int)c["init_channels"]!);
        var layers = new List<LayerBase<double>>();
        var encoder = new CamPlusPlusEncoder<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
        // SafetensorsReader keeps the fixture's float64 weights.
        var weights = SafetensorsReader.Read(Convert.FromBase64String((string)f["safetensors_base64"]!));
        var used = new HashSet<string>();
        encoder.LoadTorchWeights((name, shape) =>
        {
            // num_batches_tracked is int64.
            var values = weights.Get(name)?.DataType == "I64"
                ? new[] { (double)BitConverter.ToInt64(weights.GetRawBytes(name), 0) }
                : weights.ReadAsDouble(name);
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            used.Add(name);
            return values;
        });
        Assert.Equal(weights.Names.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        return (encoder, layers, f);
    }

    private static Tensor<double> Tensor(JToken values, params int[] shape) =>
        new(shape, new Vector<double>(values.Select(v => (double)v).ToArray()));

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task EvalEmbeddings_MatchCamPlusPlus()
    {
        await Task.Yield();
        var (encoder, _, f) = Load();
        int batch = (int)f["batch"]!, frames = (int)f["frames"]!;
        encoder.SetTrainingMode(false);
        AssertClose(f["embedding"]!, encoder.Forward(Tensor(f["features"]!, batch, frames, encoder.Configuration.FeatureDim)), "embedding");
    }

    [Fact(Timeout = 300000)]
    public async Task TrainingLossAndOneSgdStep_MatchCamPlusPlus()
    {
        await Task.Yield();
        var (encoder, layers, f) = Load();
        int batch = (int)f["batch"]!, frames = (int)f["frames"]!, e = encoder.Configuration.EmbeddingSize;
        var features = Tensor(f["features"]!, batch, frames, encoder.Configuration.FeatureDim);
        var target = Tensor(f["target"]!, batch, e);
        var engine = AiDotNetEngine.Current;
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        encoder.SetTrainingMode(true);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> loss;
        using (var tape = new GradientTape<double>())
        {
            loss = engine.ReduceSum(engine.TensorMultiply(encoder.Forward(features), target), new[] { 0, 1 }, false);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double expected = (double)f["loss"]!;
        Assert.True(Math.Abs(expected - loss[0]) <= 1e-9 * Math.Abs(expected), $"loss: reference {expected:R}, port {loss[0]:R}.");
        double rate = (double)f["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        encoder.SetTrainingMode(false);
        AssertClose(f["embedding_after_sgd"]!, encoder.Forward(features), "embedding after the step");
    }
}
