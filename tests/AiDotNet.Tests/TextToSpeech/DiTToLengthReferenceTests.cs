using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class DiTToLengthReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void CompleteCausalDecoder_MatchesTorchTransformerDecoderAndEveryGradient(int index)
    {
        using var document = Read();
        var reference = document.RootElement; var example = reference.GetProperty("cases")[index];
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => Values(example.GetProperty("gradients").GetProperty(key));
        var layers = new List<LayerBase<double>>();
        var model = new DiTToLengthPredictor<double>(AiDotNetEngine.Current, layers, Options(), new Random(7));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        model.Load((name, shape) =>
        {
            var values = Weight(name);
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        }, "");
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        static double[] Transpose(double[] values, int outputs, int inputs)
        {
            var result = new double[values.Length];
            for (int o = 0; o < outputs; o++) for (int i = 0; i < inputs; i++) result[i * outputs + o] = values[o * inputs + i];
            return result;
        }
        void Linear(DenseLayer<double> layer, string name)
        {
            int outputs = Weight(name + ".bias").Length, inputs = Weight(name + ".weight").Length / outputs;
            Close(layer.GetWeights(), Transpose(Weight(name + ".weight"), outputs, inputs), "import " + name);
            expected.Add(layer.GetWeights(), (name + ".weight", Transpose(Gradient(name + ".weight"), outputs, inputs)));
            expected.Add(layer.GetBiases(), (name + ".bias", Gradient(name + ".bias")));
        }
        void Norm(LayerNormalizationLayer<double> layer, string name)
        {
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Equal(2, owned.Count);
            Close(owned[0], Weight(name + ".weight"), "import " + name + ".weight");
            Close(owned[1], Weight(name + ".bias"), "import " + name + ".bias");
            expected.Add(owned[0], (name + ".weight", Gradient(name + ".weight")));
            expected.Add(owned[1], (name + ".bias", Gradient(name + ".bias")));
        }
        Linear(model.AudioInput, "audio_input"); Linear(model.TextInput, "text_input");
        var bos = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { model.Bos });
        Assert.Single(bos); Close(bos[0], Weight("bos"), "import bos");
        expected.Add(bos[0], ("bos", Gradient("bos")));
        for (int i = 0; i < model.Blocks.Count; i++)
        {
            var block = model.Blocks[i]; string prefix = "blocks." + i + ".";
            Linear(block.SelfQuery, prefix + "self.q"); Linear(block.SelfKey, prefix + "self.k");
            Linear(block.SelfValue, prefix + "self.v"); Linear(block.SelfOutput, prefix + "self.out");
            Linear(block.CrossQuery, prefix + "cross.q"); Linear(block.CrossKey, prefix + "cross.k");
            Linear(block.CrossValue, prefix + "cross.v"); Linear(block.CrossOutput, prefix + "cross.out");
            Norm(block.SelfNorm, prefix + "self_norm"); Norm(block.CrossNorm, prefix + "cross_norm");
            Norm(block.FeedForwardNorm, prefix + "ff_norm");
            Linear(block.FeedForward1, prefix + "ff.0"); Linear(block.FeedForward2, prefix + "ff.2");
        }
        Linear(model.Classifier, "classifier");
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(59, parameters.Count); Assert.Equal(parameters.Count, expected.Count);
        Assert.All(parameters, parameter => Assert.True(expected.ContainsKey(parameter)));
        int promptLength = example.GetProperty("prompt_length").GetInt32();
        var text = new Tensor<double>(new[] { 3, 6 }, new Vector<double>(Values(reference.GetProperty("text"))));
        var prompt = promptLength == 0 ? null : new Tensor<double>(new[] { 3, 4 },
            new Vector<double>(Values(reference.GetProperty("prompt"))));
        var labels = example.GetProperty("labels").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        Assert.Equal(labels, DiTToLengthPredictor<double>.RemainingLabels(7, promptLength));
        using var tape = new GradientTape<double>();
        var logits = model.Forward(text, prompt);
        Close(logits, Values(example.GetProperty("logits")), "complete decoder logits");
        var loss = model.CrossEntropy(logits, labels);
        Close(loss, new[] { example.GetProperty("loss").GetDouble() }, "complete decoder loss");
        var sources = parameters.Concat(new[] { text }).ToList();
        if (prompt is not null) sources.Add(prompt);
        var gradients = tape.ComputeGradients(loss, sources: sources);
        foreach (var parameter in parameters)
        {
            var wanted = expected[parameter];
            if (gradients.TryGetValue(parameter, out var actual)) Close(actual, wanted.Gradient, wanted.Name);
            else Assert.All(wanted.Gradient, value => Assert.Equal(0, value));
        }
        Close(gradients[text], Values(example.GetProperty("text_gradient")), "text gradient");
        if (prompt is not null) Close(gradients[prompt], Values(example.GetProperty("prompt_gradient")), "prompt gradient");
        Assert.Equal(example.GetProperty("expected_rounded").GetInt32(), model.ExpectedRemaining(text, prompt));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void LengthCrossEntropy_MatchesTorchForConfidentWrongPredictions(int index)
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("cross_entropy")[index];
        var (logits, labels) = LossInputs(reference);
        var model = new DiTToLengthPredictor<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), Options(2), new Random(7));
        Close(model.CrossEntropy(logits, labels), new[] { reference.GetProperty("loss").GetDouble() }, "stress cross entropy");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void LengthCrossEntropyGradient_MatchesTorchForConfidentWrongPredictions(int index)
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("cross_entropy")[index];
        var (logits, labels) = LossInputs(reference);
        var model = new DiTToLengthPredictor<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), Options(2), new Random(7));
        using var tape = new GradientTape<double>();
        var loss = model.CrossEntropy(logits, labels);
        var gradients = tape.ComputeGradients(loss, sources: new List<Tensor<double>> { logits });
        Close(gradients[logits], Values(reference.GetProperty("gradient")), "stress cross entropy gradient");
    }

    private static (Tensor<double> Logits, int[] Labels) LossInputs(JsonElement reference)
    {
        var rows = reference.GetProperty("logits");
        return (new Tensor<double>(new[] { rows.GetArrayLength(), rows[0].GetArrayLength() },
            new Vector<double>(rows.EnumerateArray().SelectMany(row => Values(row)).ToArray())),
            reference.GetProperty("labels").EnumerateArray().Select(x => x.GetInt32()).ToArray());
    }

    private static DiTToTTSOptions Options(int maximum = 8) => new()
    {
        LatentDim = 4, TextHiddenDim = 6, LengthHiddenDim = 8, LengthLayers = 2, LengthHeads = 2,
        LengthFeedForwardDim = 16, MaxLatentLength = maximum, LengthTopK = 3,
    };

    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "ditto_length_reference.json")));
    private static double[] Values(JsonElement array) => array.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"DiTTo length {name} error {error:R}.");
    }
}
