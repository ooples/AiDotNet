using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceDeepFilterReferenceTests
{
    private static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static Tensor<double> Tensor(JsonElement item) => new(
        item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
        new Vector<double>(Values(item.GetProperty("values"))));
    private static void Close(Tensor<double> actual, JsonElement expected, string name, double tolerance = 2e-9)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(), actual.Shape.ToArray());
        double error = actual.ToArray().Zip(Values(expected.GetProperty("values")), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < tolerance, name + " maximum error " + error.ToString("R"));
    }

    [Fact]
    public void OriginalLibdfFrontend_MatchesVorbisWindowSpectrumFeatureNormalizationAndSynthesis()
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_deepfilter_frontend.json")));
        var reference = document.RootElement.GetProperty("frontend");
        var widths = reference.GetProperty("erb_widths").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var frontend = new MetaVoiceDeepFilterFrontend(AiDotNetEngine.Current, widths);
        var audio = Values(reference.GetProperty("audio").GetProperty("values")).Select(x => (float)x).ToArray();
        var result = frontend.Analyze(audio);
        void FloatClose(Tensor<float> actual, string key, double tolerance)
        {
            var expected = reference.GetProperty(key); Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(), actual.Shape.ToArray());
            double error = actual.ToArray().Zip(Values(expected.GetProperty("values")), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < tolerance, key + " float32 DSP error " + error.ToString("R"));
        }
        FloatClose(result.Spectrum, "spec", 2e-8);
        FloatClose(result.Erb, "erb", 1e-5);
        FloatClose(result.Features, "features", 2e-6);
        var original = reference.GetProperty("spec");
        var sourceSpectrum = new Tensor<float>(original.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            new Vector<float>(Values(original.GetProperty("values")).Select(x => (float)x)));
        var synthesized = frontend.Synthesize(sourceSpectrum);
        double synthesisError = synthesized.Zip(Values(reference.GetProperty("synthesis").GetProperty("values")), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(synthesisError < 1e-6, "Float32 libdf synthesis error " + synthesisError.ToString("R"));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void OriginalDeepFilterNet3_MatchesEveryOutputInputGradientAndAllActiveParameters(int index)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_deepfilter_reference.json")));
        var reference = document.RootElement.GetProperty("cases")[index];
        var parameters = reference.GetProperty("parameters"); var buffers = reference.GetProperty("buffers");
        var loaded = new HashSet<string>(); var model = new MetaVoiceDeepFilterNet<double>(4, 8, 8, 8, 33, 2, 2);
        model.LoadTorch((name, shape) =>
        {
            Assert.True(loaded.Add(name), "Duplicate load: " + name);
            var entry = parameters.TryGetProperty(name, out var value) ? value : buffers.GetProperty(name);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        Assert.Equal(parameters.EnumerateObject().Count() + buffers.EnumerateObject().Count(), loaded.Count);
        var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { model });
        Assert.Equal(86, owned.Count); Assert.Equal(86, model.NamedParameters.Count);
        var spectrum = Tensor(reference.GetProperty("spec")); var erb = Tensor(reference.GetProperty("erb"));
        var features = Tensor(reference.GetProperty("features")); var engine = AiDotNetEngine.Current;
        using (var clone = (MetaVoiceDeepFilterNet<double>)model.Clone())
        {
            var restored = clone.Forward(spectrum, erb, features);
            var restoredOutputs = new[] { restored.Spectrum, restored.Mask, restored.Snr, restored.Coefficients };
            for (int i = 0; i < restoredOutputs.Length; i++)
                Close(restoredOutputs[i], reference.GetProperty("outputs")[i], "restored fitted output " + i);
        }
        using var tape = new GradientTape<double>();
        var result = model.Forward(spectrum, erb, features);
        var outputs = new[] { result.Spectrum, result.Mask, result.Snr, result.Coefficients };
        Tensor<double>? loss = null;
        for (int i = 0; i < outputs.Length; i++)
        {
            Close(outputs[i], reference.GetProperty("outputs")[i], "output " + i);
            var term = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(outputs[i], Tensor(reference.GetProperty("weights")[i])),
                Enumerable.Range(0, outputs[i].Rank).ToArray(), true), new[] { 1 });
            loss = loss is null ? term : engine.TensorAdd(loss, term);
        }
        Assert.NotNull(loss); Assert.True(Math.Abs(loss![0] - reference.GetProperty("loss").GetDouble()) < 2e-9);
        var gradients = tape.ComputeGradients(loss, sources: owned.Concat(new[] { spectrum, erb, features }).ToList());
        foreach (var pair in model.NamedParameters)
        {
            var expected = parameters.GetProperty(pair.Key).GetProperty("gradient");
            if (expected.ValueKind == JsonValueKind.Null)
            {
                Assert.StartsWith("df_dec.df_fc_a.", pair.Key);
                Assert.False(gradients.ContainsKey(pair.Value), "Author-unused parameter acquired gradient: " + pair.Key);
                continue;
            }
            Assert.True(gradients.TryGetValue(pair.Value, out var gradient), "Missing gradient: " + pair.Key);
            double error = gradient!.ToArray().Zip(Values(expected), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < 2e-9, pair.Key + " gradient error " + error.ToString("R"));
        }
        var inputs = new[] { spectrum, erb, features };
        for (int i = 0; i < inputs.Length; i++) Close(gradients[inputs[i]], reference.GetProperty("input_gradients")[i], "input gradient " + i);
    }
}
