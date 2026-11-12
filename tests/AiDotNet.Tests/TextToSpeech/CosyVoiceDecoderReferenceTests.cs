using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The Matcha-style flow estimator in CosyVoice's configuration against FunAudioLLM/CosyVoice's own
/// <c>ConditionalDecoder</c> over the Matcha-TTS components it pins (<c>tools/reference-data/cosyvoice_decoder_reference.py</c>):
/// the vector field for one input and the field after one SGD step.
/// </summary>
public class CosyVoiceDecoderReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "cosyvoice_decoder_reference.json";
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

    private static (MatchaDecoder<double> Decoder, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        var decoder = new MatchaDecoder<double>(AiDotNetEngine.Current, (int)c["in_channels"]!, (int)c["out_channels"]!,
            c["channels"]!.Select(v => (int)v).ToArray(), (double)c["dropout"]!, (int)c["attention_head_dim"]!, (int)c["num_heads"]!,
            (int)c["num_mid_blocks"]!, blocksPerLevel: (int)c["n_blocks"]!, feedForward: MatchaFeedForward.Gelu, biasMask: true);
        string path = Path.Combine(Path.GetTempPath(), $"cosyvoice_decoder_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            decoder.LoadTorchWeights((name, _) =>
            {
                used.Add(name);
                return weights[name].ToVector().Select(v => (double)v).ToArray();
            }, "");
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }
        return (decoder, f);
    }

    // [1, x; mu; spks repeated; cond, frames], as ConditionalDecoder packs them.
    private static Tensor<double> Stacked(JObject f)
    {
        int mel = (int)f["mel"]!, frames = (int)f["frames"]!;
        double[] Values(string key) => f[key]!.Select(v => (double)v).ToArray();
        var x = Values("x");
        var mu = Values("mu");
        var spks = Values("spks");
        var cond = Values("cond");
        var stacked = new Tensor<double>(new[] { 1, 4 * mel, frames });
        for (int c = 0; c < mel; c++)
            for (int t = 0; t < frames; t++)
            {
                stacked[0, c, t] = x[c * frames + t];
                stacked[0, mel + c, t] = mu[c * frames + t];
                stacked[0, 2 * mel + c, t] = spks[c];
                stacked[0, 3 * mel + c, t] = cond[c * frames + t];
            }
        return stacked;
    }

    private static void AssertField(JToken expected, Tensor<double> field, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();                                         // [1, mel, frames]
        int frames = field.Shape[0], mel = field.Shape[1];
        Assert.Equal(values.Length, field.Length);
        for (int c = 0; c < mel; c++)
            for (int t = 0; t < frames; t++)
            {
                double reference = values[c * frames + t], port = field[t, c];
                Assert.True(Math.Abs(reference - port) <= 1e-9 * Math.Max(1, Math.Abs(reference)),
                    $"{what}[{c}, {t}]: reference {reference:R}, port {port:R}.");
            }
    }

    private static Tensor<double> Ones(int frames)
    {
        var mask = new Tensor<double>(new[] { frames });
        for (int i = 0; i < frames; i++) mask[i] = 1;
        return mask;
    }

    [Fact(Timeout = 120000)]
    public async Task VectorField_MatchesCosyVoice()
    {
        await Task.Yield();
        var (decoder, f) = Load();
        AssertField(f["field"]!, decoder.EstimateChannels(Stacked(f), (double)f["t"]!, Ones((int)f["frames"]!)), "field");
    }

    [Fact(Timeout = 120000)]
    public async Task OneSgdStep_MatchesCosyVoice()
    {
        await Task.Yield();
        var (decoder, f) = Load();
        var input = Stacked(f);
        double t = (double)f["t"]!;
        var mask = Ones((int)f["frames"]!);
        var engine = AiDotNetEngine.Current;
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(decoder.Layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var field = decoder.EstimateChannels(input, t, mask);
            var loss = engine.ReduceMean(engine.TensorMultiply(field, field), new[] { 0, 1 }, false);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = (double)f["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        AssertField(f["field_after_sgd"]!, decoder.EstimateChannels(input, t, mask), "field after the step");
    }
}
