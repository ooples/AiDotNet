using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceDropoutReferenceTests
{
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void AuthorFineTuning_MatchesSevenDropoutMasksLossAllSelectedGradientsAndFreezeSelection(int index)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_dropout_reference.json")));
        var root = document.RootElement; var reference = root.GetProperty("cases")[index];
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var model = new MetaVoiceGpt<double>(AiDotNetEngine.Current, new MetaVoiceGptConfiguration
        {
            Dimension = 8, Layers = 2, Heads = 2, Context = 16, InputVocabularies = new[] { 26 },
            SwiGluMultiple = 4, SpeakerDimension = 3, SpeakerDropout = .1, Dropout = .1
        });
        var entries = reference.GetProperty("parameters");
        model.LoadTorch((name, shape) =>
        {
            if (!entries.TryGetProperty(name, out var entry)) entry = entries.GetProperty(root.GetProperty("aliases").GetProperty(name).GetString()!);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        var all = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        var remaining = entries.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);
        var expected = new Dictionary<Tensor<double>, (string Name, bool Selected, double[] Gradient)>();
        foreach (var layer in model.Layers)
        {
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Single(owned); var parameter = owned[0];
            double[] Convert(JsonElement entry, string key)
            {
                var values = Values(entry.GetProperty(key));
                if (layer is not BiasFreeLinearLayer<double>) return values;
                var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                if (shape.Length != 2) return Array.Empty<double>();
                var transposed = new double[values.Length];
                for (int o = 0; o < shape[0]; o++) for (int i = 0; i < shape[1]; i++) transposed[i * shape[0] + o] = values[o * shape[1] + i];
                return transposed;
            }
            var candidates = remaining.Where(pair => pair.Value.GetProperty("values").GetArrayLength() == parameter.Length &&
                Convert(pair.Value, "values").Zip(parameter.ToArray(), (a, b) => Math.Abs(a - b)).DefaultIfEmpty(double.PositiveInfinity).Max() < 1e-12).ToArray();
            Assert.Single(candidates); var candidate = candidates[0];
            expected.Add(parameter, (candidate.Key, candidate.Value.GetProperty("requires_grad").GetBoolean(), Convert(candidate.Value, "gradient")));
            remaining.Remove(candidate.Key);
        }
        Assert.Empty(remaining);
        var selected = reference.GetProperty("freeze").GetBoolean()
            ? AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.FineTuneLayers(reference.GetProperty("last_blocks").GetInt32()).Cast<ILayer<double>>().ToList()) : all;
        Assert.Equal(reference.GetProperty("selected_count").GetInt32(), selected.Count);
        foreach (var parameter in all) Assert.Equal(expected[parameter].Selected, selected.Contains(parameter));
        var tokens = reference.GetProperty("tokens")[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var targets = reference.GetProperty("targets")[0].EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var speaker = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { .2, -.4, .7 }));
        var masksUsed = new HashSet<string>(); var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var output = model.Forward(new[] { tokens }, speaker, training: true,
            speakerDropoutDraw: reference.GetProperty("speaker_draw").GetDouble(), dropoutMasks: (name, shape) =>
            {
                Assert.True(masksUsed.Add(name)); var mask = reference.GetProperty("masks").GetProperty(name);
                Assert.Equal(shape, mask.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
                return Values(mask.GetProperty("values"));
            });
        Assert.Equal(7, masksUsed.Count);
        Close(output.Logits[0], Values(reference.GetProperty("logits")), "logits");
        var weights = new Tensor<double>(output.Logits[0].Shape.ToArray());
        for (int i = 0; i < targets.Length; i++) weights[i, targets[i]] = -1.0 / targets.Length;
        var loss = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(engine.TensorLogSoftmax(output.Logits[0], 1), weights), new[] { 0, 1 }, true), new[] { 1 });
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "full shifted text/audio/padding CE");
        var gradients = tape.ComputeGradients(loss, sources: selected);
        foreach (var parameter in selected)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient), "Missing gradient: " + expected[parameter].Name);
            Close(gradient!, expected[parameter].Gradient, expected[parameter].Name);
        }
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"MetaVoice training dropout {name} error {error:R}.");
    }
}
