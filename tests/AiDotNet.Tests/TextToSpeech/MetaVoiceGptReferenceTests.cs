using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceGptReferenceTests
{
    [Theory]
    [InlineData(true, true, false, -1)]
    [InlineData(true, false, false, -1)]
    [InlineData(false, true, false, -1)]
    [InlineData(false, false, false, -1)]
    [InlineData(true, true, true, .05)]
    [InlineData(true, true, true, .5)]
    public void UnchangedAuthorGpt_MatchesAllLogitsIgnoredTargetObjectiveAndEveryGradient(bool causal, bool conditioned, bool training, double dropoutDraw)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_gpt_reference.json")));
        var reference = document.RootElement.GetProperty("cases").EnumerateArray().Single(x =>
            x.GetProperty("causal").GetBoolean() == causal && x.GetProperty("conditioned").GetBoolean() == conditioned &&
            x.GetProperty("training").GetBoolean() == training && (!training || x.GetProperty("dropout_draw").GetDouble() == dropoutDraw));
        var cfg = reference.GetProperty("config");
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        int[] Integers(string name) => cfg.GetProperty(name).EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var model = new MetaVoiceGpt<double>(AiDotNetEngine.Current, new MetaVoiceGptConfiguration
        {
            Dimension = 8, Layers = 2, Heads = 2, Context = 16, Causal = causal,
            InputVocabularies = Integers("vocab_sizes"), OutputVocabularies = causal ? null : Integers("target_vocab_sizes"),
            RmsNorm = causal, SwiGlu = causal, SwiGluMultiple = 4, SpeakerDimension = 3,
            SpeakerDropout = cfg.GetProperty("spkemb_dropout").GetDouble()
        });
        var entries = reference.GetProperty("parameters");
        model.LoadTorch((name, shape) =>
        {
            if (!entries.TryGetProperty(name, out var entry))
                entry = entries.GetProperty(reference.GetProperty("aliases").GetProperty(name).GetString()!);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(reference.GetProperty("unique_parameter_count").GetInt32(), parameters.Count);
        var remaining = entries.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);
        var gradientsExpected = new Dictionary<Tensor<double>, double[]>();
        var gradientIsNone = new Dictionary<Tensor<double>, bool>();
        foreach (var layer in model.Layers)
        {
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Single(owned);
            var parameter = owned[0]; var actual = parameter.ToArray();
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
            var candidates = remaining.Where(pair => pair.Value.GetProperty("values").GetArrayLength() == actual.Length &&
                Convert(pair.Value, "values").Zip(actual, (a, b) => Math.Abs(a - b)).DefaultIfEmpty(double.PositiveInfinity).Max() < 1e-12).ToArray();
            Assert.Single(candidates);
            gradientsExpected.Add(parameter, Convert(candidates[0].Value, "gradient"));
            gradientIsNone.Add(parameter, candidates[0].Value.GetProperty("gradient_is_none").GetBoolean());
            remaining.Remove(candidates[0].Key);
        }
        Assert.Empty(remaining);
        int[][] Streams(string name) => reference.GetProperty(name)[0].EnumerateArray()
            .Select(row => row.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
        var tokens = Streams("tokens"); var targets = Streams("targets");
        var speaker = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(Values(reference.GetProperty("speaker"))));
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var output = model.Forward(tokens, conditioned ? speaker : null, training: training,
            speakerDropoutDraw: training ? dropoutDraw : null);
        Close(output.Hidden, Values(reference.GetProperty("hidden")), "hidden");
        Tensor<double>? crossEntropy = null;
        for (int head = 0; head < output.Logits.Length; head++)
        {
            var logits = output.Logits[head]; Close(logits, Values(reference.GetProperty("logits")[head]), "head " + head);
            var weights = new Tensor<double>(logits.Shape.ToArray());
            int count = targets[head].Count(id => id >= 0);
            for (int t = 0; t < targets[head].Length; t++)
                if (targets[head][t] >= 0) weights[t, targets[head][t]] = -1.0 / (count * output.Logits.Length);
            var term = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(engine.TensorLogSoftmax(logits, 1), weights), new[] { 0, 1 }, true), new[] { 1 });
            crossEntropy = crossEntropy is null ? term : engine.TensorAdd(crossEntropy, term);
        }
        Close(crossEntropy!, new[] { reference.GetProperty("cross_entropy").GetDouble() }, "mean head CE with ignored text targets");
        var hiddenWeights = new Tensor<double>(output.Hidden.Shape.ToArray(), new Vector<double>(Values(reference.GetProperty("hidden_weights"))));
        var hiddenObjective = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(output.Hidden, hiddenWeights), new[] { 0, 1 }, true), new[] { 1 });
        var loss = engine.TensorAdd(crossEntropy!, engine.TensorMultiplyScalar(hiddenObjective, .07));
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { speaker }).ToList());
        foreach (var parameter in parameters)
        {
            if (gradients.TryGetValue(parameter, out var gradient)) Close(gradient, gradientsExpected[parameter], "parameter gradient");
            else
            {
                Assert.True(gradientIsNone[parameter], "An active author parameter has no native gradient.");
                Assert.All(gradientsExpected[parameter], value => Assert.Equal(0, value));
            }
        }
        if (gradients.TryGetValue(speaker, out var speakerGradient))
            Close(speakerGradient, Values(reference.GetProperty("speaker_gradient")), "speaker gradient");
        else Assert.All(Values(reference.GetProperty("speaker_gradient")), value => Assert.Equal(0, value));
        using var noGrad = new NoGradScope<double>();
        var inference = model.Forward(tokens, conditioned ? speaker : null, lastOnly: causal);
        for (int head = 0; head < inference.Logits.Length; head++)
            Close(inference.Logits[head], Values(reference.GetProperty("inference_logits")[head]), "inference head " + head);
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"MetaVoice {name} maximum error {error:R}.");
    }
}
