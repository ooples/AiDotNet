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
/// CosyVoice 1's language model against FunAudioLLM/CosyVoice's own <c>TransformerLM</c>
/// (<c>tools/reference-data/cosyvoice_lm_reference.py</c>): the LM input, its log-probabilities, the training loss and
/// the loss after one SGD step; and the repetition-aware sampler's rules.
/// </summary>
public class CosyVoiceLmReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "cosyvoice_lm_reference.json";
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

    private static (CosyVoiceLm<double> Lm, List<LayerBase<double>> Layers, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        var te = f["text_encoder"]!;
        var l = f["llm"]!;
        int textInput = (int)c["text_encoder_input_size"]!, llm = (int)c["llm_input_size"]!;
        var configuration = new CosyVoiceLmConfiguration(
            TextVocabulary: (int)c["text_token_size"]!, SpeechVocabulary: (int)c["speech_token_size"]!, TextEncoderInputSize: textInput,
            LlmSize: llm, SpeakerDim: (int)c["spk_embed_dim"]!,
            TextEncoder: new WenetEncoderConfiguration(textInput, (int)te["output_size"]!, (int)te["attention_heads"]!, (int)te["linear_units"]!,
                (int)te["num_blocks"]!, WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: 1),
            Llm: new WenetEncoderConfiguration(llm, (int)l["output_size"]!, (int)l["attention_heads"]!, (int)l["linear_units"]!,
                (int)l["num_blocks"]!, WenetInputLayer.LinearLegacy, Swish: false, Conformer: false, StaticChunkSize: 1));
        var layers = new List<LayerBase<double>>();
        var lm = new CosyVoiceLm<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
        string path = Path.Combine(Path.GetTempPath(), $"cosyvoice_lm_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            lm.LoadTorchWeights((name, shape) =>
            {
                var tensor = weights[name];
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            }, "");
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }
        return (lm, layers, f);
    }

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static Tensor<double> XVector(JObject f) =>
        new(new[] { f["xvector"]!.Count() }, new Vector<double>(f["xvector"]!.Select(v => (double)v).ToArray()));

    [Fact(Timeout = 120000)]
    public async Task InputLogProbabilitiesAndLoss_MatchCosyVoice()
    {
        await Task.Yield();
        var (lm, _, f) = Load();
        var text = Ints(f["text"]!);
        var speech = Ints(f["speech"]!);
        var xvector = XVector(f);
        var random = new Random(0);
        var input = lm.Input(text, speech, xvector, false, random);
        AssertClose(f["lm_input"]!, input, "LM input");
        AssertClose(f["logp"]!, lm.LogProbabilities(input, false, random), "log-probabilities");
        double expected = (double)f["loss"]!, actual = lm.Loss(text, speech, xvector, false, random)[0];
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * expected, $"loss: reference {expected:R}, port {actual:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task OneSgdStep_MatchesCosyVoice()
    {
        await Task.Yield();
        var (lm, layers, f) = Load();
        var text = Ints(f["text"]!);
        var speech = Ints(f["speech"]!);
        var xvector = XVector(f);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var loss = lm.Loss(text, speech, xvector, false, new Random(0));
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = (double)f["learning_rate"]!;
        var engine = AiDotNetEngine.Current;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double expected = (double)f["loss_after_sgd"]!, actual = lm.Loss(text, speech, xvector, false, new Random(0))[0];
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * expected, $"loss after the step: reference {expected:R}, port {actual:R}.");
    }

    [Fact(Timeout = 60000)]
    public async Task RepetitionAwareSampling_UsesTheOriginal300MFallbackRule()
    {
        await Task.Yield();
        var c = new CosyVoiceLmConfiguration(10, 6, 4, 4, 2, new WenetEncoderConfiguration(4, 4, 1, 4, 1, WenetInputLayer.Linear, true, true),
            new WenetEncoderConfiguration(4, 4, 1, 4, 1, WenetInputLayer.LinearLegacy, false, false));
        var logp = new[] { Math.Log(0.9), Math.Log(0.04), Math.Log(0.03), Math.Log(0.02), Math.Log(0.01) };
        var original = JObject.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "cosyvoice_sampling_reference.json")))["sources"]![0]!["cases"]!;
        Assert.Equal(original[0]!["token"]!.Value<int>(),
            CosyVoiceLm<double>.SampleRepetitionAware(logp, new[] { 1, 2 }, c, new FixedSamplingRandom(.5)));
        // The original author fallback may select the repeated token again.
        Assert.Equal(original[1]!["token"]!.Value<int>(),
            CosyVoiceLm<double>.SampleRepetitionAware(logp, new[] { 0, 3 }, c, new FixedSamplingRandom(.5)));
        Assert.Equal(original[2]!["token"]!.Value<int>(),
            CosyVoiceLm<double>.SampleRepetitionAware(logp, new[] { 0, 3 }, c, new FixedSamplingRandom(.95)));
        var old = new[] { 0 }.Concat(Enumerable.Repeat(4, 10)).ToArray();
        Assert.Equal(original[3]!["token"]!.Value<int>(),
            CosyVoiceLm<double>.SampleRepetitionAware(logp, old, c, new FixedSamplingRandom(.5)));
    }

    private sealed class FixedSamplingRandom : Random
    {
        private readonly double _value;
        internal FixedSamplingRandom(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
