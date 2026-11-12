using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceIpaPriorTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OriginalIpaTransformer_MatchesAll39GradientsIncludingObservedUnmodifiedTrainingDropout(bool training)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "openvoice_ipa_prior_reference.json")));
        var reference = document.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("training").GetBoolean() == training);
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var entries = reference.GetProperty("parameters"); var loaded = new HashSet<string>();
        var prior = new OpenVoiceIpaPrior<double>(AiDotNetEngine.Current, 17, 8, 4, 16, 2, 2, 3, .1);
        prior.LoadTorch((name, shape) =>
        {
            Assert.True(loaded.Add(name)); var entry = entries.GetProperty(name);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        Assert.Equal(39, loaded.Count);
        double[] Convert(string name, JsonElement entry, string key)
        {
            var values = Values(entry.GetProperty(key));
            if (!name.StartsWith("encoder.attn_layers.") || !name.Contains(".conv_") || !name.EndsWith(".weight")) return values;
            var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            var transposed = new double[values.Length];
            for (int output = 0; output < shape[0]; output++) for (int input = 0; input < shape[1]; input++)
                transposed[input * shape[0] + output] = values[output * shape[1] + input];
            return transposed;
        }
        var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(prior.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(39, owned.Count);
        var remaining = entries.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        foreach (var parameter in owned)
        {
            var candidates = remaining.Where(x => x.Value.GetProperty("values").GetArrayLength() == parameter.Length &&
                Convert(x.Key, x.Value, "values").Zip(parameter.ToArray(), (a, b) => Math.Abs(a - b)).Max() < 1e-12).ToArray();
            Assert.Single(candidates); var selected = candidates[0];
            expected.Add(parameter, (selected.Key, Convert(selected.Key, selected.Value, "gradient"))); remaining.Remove(selected.Key);
        }
        Assert.Empty(remaining);
        var tokens = Values(reference.GetProperty("tokens")); int time = tokens.Length, draws = 0;
        double[] Masks(string site, int[] shape)
        {
            draws++; var parts = site.Split('.'); int block = int.Parse(parts[1]);
            int ordinal = parts[2] switch { "attention_probs" => 0, "attention_output" => 1, "feed_forward_hidden" => 2, "feed_forward_output" => 3, _ => throw new InvalidDataException(site) };
            var mask = reference.GetProperty("masks")[4 * block + ordinal]; var values = Values(mask.GetProperty("values"));
            if (ordinal == 0)
            {
                Assert.Equal(new[] { time, time }, shape);
                return values.Skip(int.Parse(parts[3]) * time * time).Take(time * time).ToArray();
            }
            if (ordinal == 2) { Assert.Equal(new[] { 1, 16, time }, shape); return values; }
            Assert.Equal(new[] { time, 8 }, shape); var rows = new double[values.Length];
            for (int t = 0; t < time; t++) for (int c = 0; c < 8; c++) rows[t * 8 + c] = values[c * time + t];
            return rows;
        }
        var engine = AiDotNetEngine.Current; using var tape = new GradientTape<double>();
        var result = prior.Forward(new Tensor<double>(new[] { time }, new Vector<double>(tokens)), training, Masks);
        var outputs = new[] { result.Hidden, result.Mean, result.LogScale }; Tensor<double>? loss = null;
        for (int index = 0; index < outputs.Length; index++)
        {
            double error = outputs[index].ToArray().Zip(Values(reference.GetProperty("outputs")[index]), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 2e-9, "Original prior output error " + error.ToString("R"));
            var probe = new Tensor<double>(outputs[index].Shape.ToArray(), new Vector<double>(Values(reference.GetProperty("probes")[index])));
            var term = engine.TensorMultiplyScalar(engine.Reshape(engine.ReduceSum(engine.TensorMultiply(outputs[index], probe), new[] { 0, 1, 2 }, true), new[] { 1 }), (index + 1) / 3.0);
            loss = loss is null ? term : engine.TensorAdd(loss, term);
        }
        Assert.Equal(training ? 10 : 0, draws); Assert.True(Math.Abs(loss![0] - reference.GetProperty("loss").GetDouble()) < 2e-9);
        var gradients = tape.ComputeGradients(loss, sources: owned);
        foreach (var parameter in owned)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient), "Missing prior gradient: " + expected[parameter].Name);
            double error = gradient!.ToArray().Zip(expected[parameter].Gradient, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 2e-9, expected[parameter].Name + " gradient error " + error.ToString("R"));
        }
    }
}
