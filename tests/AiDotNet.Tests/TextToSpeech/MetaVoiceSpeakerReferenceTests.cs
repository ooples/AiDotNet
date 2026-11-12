using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceSpeakerReferenceTests
{
    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "metavoice_speaker_reference.json")));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void AuthorSpeakerForward_MatchesAll14IndependentParametersAndMelGradients(int batch)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = Reference();
        var reference = document.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("batch").GetInt32() == batch);
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var model = new MetaVoiceSpeakerEncoder<double>(AiDotNetEngine.Current, 5, 8, 6);
        var entries = reference.GetProperty("parameters"); var loaded = new HashSet<string>();
        model.LoadTorch((name, shape) =>
        {
            Assert.True(loaded.Add(name)); var entry = entries.GetProperty(name);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        Assert.Equal(14, loaded.Count);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(14, parameters.Count);
        var remaining = entries.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        foreach (var parameter in parameters)
        {
            double[] Convert(JsonElement entry, string key)
            {
                var values = Values(entry.GetProperty(key));
                var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                if (shape.Length != 2) return values;
                var transposed = new double[values.Length];
                for (int o = 0; o < shape[0]; o++) for (int i = 0; i < shape[1]; i++) transposed[i * shape[0] + o] = values[o * shape[1] + i];
                return transposed;
            }
            var candidates = remaining.Where(pair => pair.Value.GetProperty("values").GetArrayLength() == parameter.Length &&
                Convert(pair.Value, "values").Zip(parameter.ToArray(), (a, b) => Math.Abs(a - b)).Max() < 1e-12).ToArray();
            Assert.Single(candidates); var candidate = candidates[0];
            expected.Add(parameter, (candidate.Key, Convert(candidate.Value, "gradient"))); remaining.Remove(candidate.Key);
        }
        Assert.Empty(remaining);
        int frames = reference.GetProperty("frames").GetInt32();
        var mels = new Tensor<double>(new[] { batch, frames, 5 }, new Vector<double>(Values(reference.GetProperty("mels"))));
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var output = model.ForwardMels(mels);
        Close(output, Values(reference.GetProperty("output")), "embedding");
        var weights = new Tensor<double>(output.Shape.ToArray(), new Vector<double>(Values(reference.GetProperty("weights"))));
        var loss = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(output, weights), new[] { 0, 1 }, true), new[] { 1 });
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { mels }).ToList());
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient), "Missing gradient: " + expected[parameter].Name);
            Close(gradient!, expected[parameter].Gradient, expected[parameter].Name);
        }
        Close(gradients[mels], Values(reference.GetProperty("gradient")), "mel gradient");
    }

    [Fact]
    public void OriginalLibrosaFrontend_MatchesConstantPaddingAndRawPartialPreparation()
    {
        AiDotNetEngine.ResetToCpu();
        using var document = Reference();
        var model = new MetaVoiceSpeakerEncoder<double>(AiDotNetEngine.Current);
        foreach (var item in document.RootElement.GetProperty("frontends").EnumerateArray())
        {
            var audio = item.GetProperty("audio").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            var actual = model.PreparePartials(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)));
            Assert.Equal(item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()), actual.Shape.ToArray());
            double error = actual.ToArray().Zip(item.GetProperty("partials").EnumerateArray().Select(x => x.GetDouble()), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 2e-6, $"Original float32 Librosa power-mel error {error:R} for {audio.Length} samples.");
        }
    }

    [Fact]
    public void AuthorPartialSlices_MatchCoverageAndRawWaveformPaddingBoundaries()
    {
        using var document = Reference();
        foreach (var item in document.RootElement.GetProperty("boundaries").EnumerateArray())
        {
            var starts = MetaVoiceSpeakerEncoder<double>.PartialStarts(item.GetProperty("samples").GetInt32());
            Assert.Equal(item.GetProperty("starts").EnumerateArray().Select(x => x.GetInt32()), starts);
            Assert.Equal(item.GetProperty("stops").EnumerateArray().Select(x => x.GetInt32()), starts.Select(x => (x + 160) * 160));
        }
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"MetaVoice speaker {name} error {error:R}.");
    }
}
