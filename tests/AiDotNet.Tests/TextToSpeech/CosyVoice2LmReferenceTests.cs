using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice 2's language model against FunAudioLLM/CosyVoice's <c>Qwen2LM</c> over Hugging Face's Qwen2
/// (<c>tools/reference-data/cosyvoice2_lm_reference.py</c>): the unistream log-probabilities, the loss of the unistream
/// and both bistream forms, and the loss after one SGD step.
/// </summary>
public class CosyVoice2LmReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string fileName = "cosyvoice2_lm_reference.json";
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
    });

    /// <summary>The reference script's weights(): exact integer arithmetic on the parameter's name and index.</summary>
    private static double[] Weights(string name, int count)
    {
        long h = 0;
        for (int j = 0; j < name.Length; j++) h += (j + 1) * (long)name[j];
        h %= 100003;
        double b = name.Contains("norm") ? 1.0 : 0.0;
        var values = new double[count];
        for (long i = 0; i < count; i++) values[i] = b + 0.1 * (((i * 7919 + h) % 2003) / 2003.0 - 0.5);
        return values;
    }

    // A Random whose draws are all `value`: it pins Qwen2LM's bistream/unistream choice as the script pins random.random.
    private sealed class Fixed : Random
    {
        private readonly double _value;
        public Fixed(double value) => _value = value;
        protected override double Sample() => _value;
    }

    private static (CosyVoice2Lm<double> Lm, List<LayerBase<double>> Layers) Build()
    {
        var q = Data.Value["qwen"]!;
        var configuration = new CosyVoice2LmConfiguration(new LlamaConfiguration
        {
            VocabSize = (int)q["vocab_size"]!, Hidden = (int)q["hidden_size"]!, Layers = (int)q["num_hidden_layers"]!,
            Heads = (int)q["num_attention_heads"]!, KeyValueHeads = (int)q["num_key_value_heads"]!,
            Intermediate = (int)q["intermediate_size"]!, RmsEpsilon = (double)q["rms_norm_eps"]!, RopeTheta = (double)q["rope_theta"]!,
            AttentionBias = true, IncludeHead = false,
        }, (int)Data.Value["speech_tokens"]!);
        var layers = new List<LayerBase<double>>();
        var lm = new CosyVoice2Lm<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
        var expected = Data.Value["parameters"]!.ToDictionary(p => (string)p[0]!, p => p[1]!.Select(s => (int)s).ToArray());
        var used = new HashSet<string>();
        lm.LoadTorchWeights((name, shape) =>
        {
            Assert.True(expected.TryGetValue(name, out var reference), $"{name} is not a Qwen2LM parameter.");
            Assert.Equal(reference.Aggregate(1, (a, s) => a * s), shape.Aggregate(1, (a, s) => a * s));
            used.Add(name);
            return Weights(name, shape.Aggregate(1, (a, s) => a * s));
        }, "");
        Assert.Equal(expected.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        return (lm, layers);
    }

    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static void AssertClose(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: reference {expected:R}, port {actual:R}.");

    [Fact(Timeout = 300000)]
    public async Task Losses_MatchQwen2LM_InEverySequenceForm()
    {
        await Task.Yield();
        var (lm, _) = Build();
        var losses = Data.Value["losses"]!;
        foreach (var c in Data.Value["cases"]!)
        {
            string name = (string)c["name"]!;
            var draw = new Fixed(name == "unistream" ? 0.9 : 0.1);
            using var _ = new NoGradScope<double>();
            AssertClose((double)losses[name]!, lm.Loss(Ints(c["text"]!), Ints(c["speech"]!), draw)[0], name);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task UnistreamLogProbabilitiesAndOneSgdStep_MatchQwen2LM()
    {
        await Task.Yield();
        var (lm, layers) = Build();
        var c = Data.Value["cases"]![0]!;
        var text = Ints(c["text"]!);
        var speech = Ints(c["speech"]!);
        var engine = AiDotNetEngine.Current;
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> loss;
        using (var tape = new GradientTape<double>())
        {
            loss = lm.Loss(text, speech, new Fixed(0.9));
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose((double)Data.Value["losses"]!["unistream"]!, loss[0], "loss");

        using (new NoGradScope<double>())
        {
            var (input, _) = lm.TrainingSequence(text, speech, new Fixed(0.9));
            var logp = engine.TensorLogSoftmax(lm.LogitsOf(input), axis: -1);
            var expected = Data.Value["logp"]!.Select(v => (double)v).ToArray();
            Assert.Equal(expected.Length, logp.Length);
            for (int i = 0; i < expected.Length; i++) AssertClose(expected[i], logp[i], $"logp[{i}]");
        }

        double rate = (double)Data.Value["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        using (new NoGradScope<double>())
            AssertClose((double)Data.Value["loss_after_sgd"]!, lm.Loss(text, speech, new Fixed(0.9))[0], "loss after the step");
    }
}
