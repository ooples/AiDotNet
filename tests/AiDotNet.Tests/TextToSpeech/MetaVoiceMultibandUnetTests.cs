using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceMultibandUnetTests
{
    [Theory]
    [InlineData(1, 1031)]
    [InlineData(2, 1027)]
    public void UnchangedAuthorDenoiser_MatchesOddPaddingCodecConditioningAndAll158Gradients(int batch, int length)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_multiband_unet.json")));
        var reference = document.RootElement.GetProperty("cases").EnumerateArray()
            .Single(x => x.GetProperty("batch").GetInt32() == batch);
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var model = new MetaVoiceDiffusionUnet<double>(AiDotNetEngine.Current, hidden: 4, growth: 2,
            codecDimension: 3, steps: 32);
        var entries = reference.GetProperty("parameters");
        var loaded = new HashSet<string>();
        model.LoadTorch((name, shape) =>
        {
            Assert.True(loaded.Add(name), "Duplicate imported parameter: " + name);
            var entry = entries.GetProperty(name);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        Assert.Equal(158, loaded.Count);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(158, parameters.Count);
        var remaining = entries.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        foreach (var parameter in parameters)
        {
            var values = parameter.ToArray();
            var candidates = remaining.Where(pair => pair.Value.GetProperty("values").GetArrayLength() == values.Length &&
                Values(pair.Value.GetProperty("values")).Zip(values, (a, b) => Math.Abs(a - b)).Max() < 1e-12).ToArray();
            Assert.Single(candidates);
            var candidate = candidates[0];
            expected.Add(parameter, (candidate.Key, Values(candidate.Value.GetProperty("gradient"))));
            remaining.Remove(candidate.Key);
        }
        Assert.Empty(remaining);
        Tensor<double> Tensor(string key, int[] shape) => new(shape, new Vector<double>(Values(reference.GetProperty(key))));
        var waveform = Tensor("waveform", new[] { batch, 1, length });
        var condition = Tensor("condition", new[] { batch, 3, 7 });
        var steps = reference.GetProperty("step").ValueKind == JsonValueKind.Array
            ? reference.GetProperty("step").EnumerateArray().Select(x => x.GetInt32()).ToArray()
            : Enumerable.Repeat(reference.GetProperty("step").GetInt32(), batch).ToArray();
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var output = model.Forward(waveform, steps, condition);
        Close(output, Values(reference.GetProperty("output")), "waveform");
        var weighted = engine.ReduceSum(engine.TensorMultiply(output, Tensor("weights", output.Shape.ToArray())), new[] { 0, 1, 2 }, true);
        var square = engine.ReduceMean(engine.TensorMultiply(output, output), new[] { 0, 1, 2 }, true);
        var loss = engine.Reshape(engine.TensorAdd(weighted, engine.TensorMultiplyScalar(square, .13)), new[] { 1 });
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { waveform, condition }).ToList());
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient), "Missing gradient for " + expected[parameter].Name);
            Close(gradient!, expected[parameter].Gradient, expected[parameter].Name);
        }
        Close(gradients[waveform], Values(reference.GetProperty("waveform_gradient")), "audio gradient");
        Close(gradients[condition], Values(reference.GetProperty("condition_gradient")), "codec conditioning gradient");
        Assert.True(gradients[condition].ToArray().Max(Math.Abs) > 0, "Conditioning must reach the deepest skip.");
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"AudioCraft denoiser {name} maximum error {error:R}.");
    }
}
