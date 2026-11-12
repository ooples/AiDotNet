using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// LLaMA-Omni's speech adaptor and CTC speech decoder against the release's <c>EncoderProjectorConcat</c> and
/// <c>SpeechGeneratorCTC</c> (<c>tools/reference-data/llamaomni_reference.py</c>): the adaptor's output, the CTC loss of
/// two utterances, the prediction, and the loss after one SGD step; and the released modules (gated on LLAMAOMNI_SHARD,
/// the path of <c>model-00004-of-00004.safetensors</c>).
/// </summary>
public class LlamaOmniModulesReferenceTests
{
    private static JObject Fixture(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static Dictionary<string, double[]> Safetensors(string base64)
    {
        string path = Path.Combine(Path.GetTempPath(), $"llamaomni_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String(base64));
        try
        {
            return new WeightLoader().LoadWeights(path).ToDictionary(p => p.Key, p => p.Value.ToVector().Select(v => (double)v).ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Tensor<double> Values(JToken t, params int[] shape) => new(shape, new Vector<double>(t.Select(v => (double)v).ToArray()));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(values[i])), $"{what}[{i}]: {values[i]:R} vs {actual[i]:R}");
    }

    private static (LlamaOmniAdaptor<double> Adaptor, LlamaOmniSpeechDecoder<double> Decoder, List<LayerBase<double>> DecoderLayers) Tiny(JObject f)
    {
        var engine = AiDotNetEngine.Current;
        var adaptor = new LlamaOmniAdaptor<double>(engine, new List<LayerBase<double>>(), 6, 5, 2048, 16, new Random(0));
        var layers = new List<LayerBase<double>>();
        var decoder = new LlamaOmniSpeechDecoder<double>(engine, layers,
            new LlamaOmniDecoderConfiguration(16, 2, 8, 2, 16, 3, 12, 1e-5, 500000, 8, 1, 4, 32), new Random(0));
        var projector = Safetensors((string)f["projector_base64"]!);
        var generator = Safetensors((string)f["generator_base64"]!);
        var used = new HashSet<string>();
        adaptor.Load((name, _) => { used.Add(name); return projector[name]; }, "");
        decoder.Load((name, _) => { used.Add(name); return generator[name]; }, "");
        Assert.All(projector.Keys.Concat(generator.Keys), k => Assert.Contains(k, used));
        return (adaptor, decoder, layers);
    }

    private static List<(Tensor<double>, int[])> Utterances(JObject f)
    {
        // The release reads the hidden states at the label positions and the units up to the IGNORE padding.
        var labels = f["labels"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var units = f["units"]!.Select(r => r.Select(v => (int)v).Where(v => v >= 0).ToArray()).ToArray();
        var reps = new[] { Values(f["reps0"]!, 7, 16), Values(f["reps1"]!, 5, 16) };
        var engine = AiDotNetEngine.Current;
        return Enumerable.Range(0, 2).Select(n =>
        {
            var rows = Enumerable.Range(0, labels[n].Length).Where(i => i < reps[n].Shape[0] && labels[n][i] != -100).ToArray();
            return (engine.TensorIndexSelect(reps[n], new Tensor<int>(new[] { rows.Length }, new Vector<int>(rows)), 0), units[n]);
        }).ToList();
    }

    [Fact(Timeout = 120000)]
    public async Task TinyModules_MatchRelease()
    {
        await Task.Yield();
        var f = Fixture("llamaomni_reference.json");
        var (adaptor, decoder, layers) = Tiny(f);
        using (new NoGradScope<double>())
        {
            Close(f["adapted"]!, adaptor.Forward(Values(f["encoder"]!, 23, 6)), "adapted", 1e-9);
            Assert.Equal(f["prediction"]!.Select(v => (int)v).ToArray(), decoder.Alignment(Values(f["reps0"]!, 7, 16)));
        }
        var utterances = Utterances(f);
        double loss = decoder.Loss(utterances)[0];
        Assert.True(Math.Abs((double)f["loss"]! - loss) <= 1e-9 * Math.Abs((double)f["loss"]!), $"CTC loss: {(double)f["loss"]!:R} vs {loss:R}");

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>()) grads = tape.ComputeGradients(decoder.Loss(utterances), sources: parameters);
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        double after = decoder.Loss(utterances)[0];
        Assert.True(Math.Abs((double)f["loss_after_sgd"]! - after) <= 1e-9 * Math.Abs((double)f["loss_after_sgd"]!), $"loss after SGD: {(double)f["loss_after_sgd"]!:R} vs {after:R}");
    }

    [Fact(Timeout = 1800000)]
    public async Task ReleasedModules_MatchRelease()
    {
        await Task.Yield();
        string? shard = Environment.GetEnvironmentVariable("LLAMAOMNI_SHARD");
        if (shard is null || !File.Exists(shard)) return;
        var f = Fixture("llamaomni_realweights_reference.json");
        var engine = AiDotNetEngine.Current;
        var weights = new WeightLoader().LoadWeights(shard);
        double[] Read(string name) => weights[name].ToVector().Select(v => (double)v).ToArray();
        var adaptor = new LlamaOmniAdaptor<double>(engine, new List<LayerBase<double>>(), 1280, 5, 2048, 4096, new Random(0));
        adaptor.Load((name, _) => Read(name), "model.speech_projector.");
        var decoder = new LlamaOmniSpeechDecoder<double>(engine, new List<LayerBase<double>>(),
            new LlamaOmniDecoderConfiguration(4096, 2, 4096, 32, 11008, 25, 1000, 1e-5, 500000, 8, 1, 4, 8192), new Random(0));
        decoder.Load((name, _) => Read(name), "speech_generator.");
        weights.Clear();
        using var _ = new NoGradScope<double>();
        Close(f["adapted"]!, adaptor.Forward(Values(f["encoder"]!, 50, 1280)), "adapted", 1e-8);
        var reps = Values(f["reps"]!, 6, 4096);
        var units = f["units"]!.Select(v => (int)v).ToArray();
        double loss = decoder.Loss(new List<(Tensor<double>, int[])> { (reps, units) })[0];
        Assert.True(Math.Abs((double)f["loss"]! - loss) <= 1e-8 * Math.Abs((double)f["loss"]!), $"CTC loss: {(double)f["loss"]!:R} vs {loss:R}");
        Assert.Equal(f["prediction"]!.Select(v => (int)v).ToArray(), decoder.Alignment(reps));
    }
}
