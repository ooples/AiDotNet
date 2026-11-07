using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The wenet encoder with ESPnet relative positions against FunAudioLLM/CosyVoice's own <c>ConformerEncoder</c> and
/// <c>TransformerEncoder</c> (<c>tools/reference-data/wenet_encoder_reference.py</c>), in CosyVoice 1's three
/// configurations (causal Conformer text encoder, causal Transformer language model with the legacy input layer,
/// full-attention Conformer flow encoder), and one SGD step.
/// </summary>
public class WenetEncoderReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "wenet_encoder_reference.json";
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

    private static List<(WenetEncoder<double> Encoder, List<LayerBase<double>> Layers)> Load(JObject f)
    {
        string path = Path.Combine(Path.GetTempPath(), $"wenet_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            var result = new List<(WenetEncoder<double>, List<LayerBase<double>>)>();
            foreach (var spec in f["encoders"]!)
            {
                bool conformer = (string)spec["kind"]! == "conformer";
                var configuration = new WenetEncoderConfiguration(
                    InputSize: (int)spec["input_size"]!, OutputSize: (int)spec["output_size"]!, Heads: (int)spec["attention_heads"]!,
                    LinearUnits: (int)spec["linear_units"]!, Blocks: (int)spec["num_blocks"]!,
                    InputLayer: (string)spec["input_layer"]! == "linear_legacy" ? WenetInputLayer.LinearLegacy : WenetInputLayer.Linear,
                    Swish: conformer, Conformer: conformer, StaticChunkSize: (int)spec["static_chunk_size"]!);
                var layers = new List<LayerBase<double>>();
                var encoder = new WenetEncoder<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
                string prefix = (string)spec["name"]! + ".";
                encoder.LoadTorchWeights((name, shape) =>
                {
                    var tensor = weights[name];
                    Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                    used.Add(name);
                    return tensor.ToVector().Select(v => (double)v).ToArray();
                }, prefix);
                result.Add((encoder, layers));
            }
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
            return result;
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Tensor<double> Input(JObject f, int index, int size)
    {
        var values = f["inputs"]![index]!.Select(v => (double)v).ToArray();
        return new Tensor<double>(new[] { values.Length / size, size }, new Vector<double>(values));
    }

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task EveryConfiguration_MatchesCosyVoice()
    {
        await Task.Yield();
        var f = Reference();
        var encoders = Load(f);
        for (int i = 0; i < encoders.Count; i++)
        {
            var encoder = encoders[i].Encoder;
            var output = encoder.Forward(Input(f, i, encoder.Configuration.InputSize), training: false, new Random(0));
            AssertClose(f["outputs"]![i]!, output, (string)f["encoders"]![i]!["name"]!);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task OneSgdStep_MatchesCosyVoice()
    {
        await Task.Yield();
        var f = Reference();
        var (encoder, layers) = Load(f)[0];
        var input = Input(f, 0, encoder.Configuration.InputSize);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        var engine = AiDotNetEngine.Current;
        using (var tape = new GradientTape<double>())
        {
            var output = encoder.Forward(input, training: false, new Random(0));
            var loss = engine.ReduceMean(engine.TensorMultiply(output, output), new[] { 0, 1 }, false);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = (double)f["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        AssertClose(f["after_sgd"]!, encoder.Forward(input, training: false, new Random(0)), "after the step");
    }
}
