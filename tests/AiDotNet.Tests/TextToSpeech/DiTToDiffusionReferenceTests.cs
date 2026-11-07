using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class DiTToDiffusionReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void CompleteConditionalAndUnconditionalField_MatchesIndependentTorchAndEveryGradient(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "ditto_diffusion_reference.json")));
        var reference = document.RootElement; var example = reference.GetProperty("cases")[index];
        double[] Values(JsonElement values) => values.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => Values(example.GetProperty("gradients").GetProperty(key));
        Close(DiTToDiffusionTransformer<double>.TimestepEmbedding<double>(.37, 256),
            Values(reference.GetProperty("double_operator_timestep")), "original DiT timestep with double operator precision");
        var layers = new List<LayerBase<double>>(); var engine = AiDotNetEngine.Current;
        var model = new DiTToDiffusionTransformer<double>(engine, layers, new DiTToTTSOptions
        {
            LatentDim = 4, DiTHiddenDim = 8, DiTHeads = 2, DiTLayers = 2, TextHiddenDim = 6, DiTFeedForwardMultiplier = 2
        }, new Random(17));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        model.Load((name, shape) =>
        {
            var values = Weight(name);
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        }, "");
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        static double[] Transpose(double[] values, int output, int input)
        {
            var result = new double[values.Length];
            for (int o = 0; o < output; o++) for (int i = 0; i < input; i++) result[i * output + o] = values[o * input + i];
            return result;
        }
        void Linear(DenseLayer<double> layer, string name)
        {
            int outputs = Weight(name + ".bias").Length, inputs = Weight(name + ".weight").Length / outputs;
            Close(layer.GetWeights(), Transpose(Weight(name + ".weight"), outputs, inputs), "import " + name);
            expected.Add(layer.GetWeights(), (name + ".weight", Transpose(Gradient(name + ".weight"), outputs, inputs)));
            expected.Add(layer.GetBiases(), (name + ".bias", Gradient(name + ".bias")));
        }
        void Value(BiasFreeLinearLayer<double> layer, string name)
        {
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Single(owned); Close(owned[0], Weight(name), "import " + name);
            expected.Add(owned[0], (name, Gradient(name)));
        }
        Linear(model.Input, "input"); Linear(model.TextKeys, "text_keys"); Linear(model.TimeIn, "time.0"); Linear(model.TimeOut, "time.2");
        Linear(model.TextConditionIn, "text_condition.0"); Linear(model.TextConditionOut, "text_condition.2");
        Linear(model.SharedModulation, "shared_adaln"); Value(model.NullText, "null_text");
        Linear(model.SkipIn, "long_skip.0"); Linear(model.SkipOut, "long_skip.2"); Linear(model.Output, "output");
        for (int i = 0; i < model.Blocks.Count; i++)
        {
            var block = model.Blocks[i]; string name = "blocks." + i;
            Value(block.ModulationOffset, name + ".adaln_offset");
            Linear(block.SelfQuery, name + ".self.q"); Linear(block.SelfKey, name + ".self.k");
            Linear(block.SelfValue, name + ".self.v"); Linear(block.SelfOutput, name + ".self.out");
            Linear(block.CrossQuery, name + ".cross.q"); Linear(block.CrossKey, name + ".cross.k");
            Linear(block.CrossValue, name + ".cross.v"); Linear(block.CrossOutput, name + ".cross.out");
            Linear(block.GegluIn, name + ".geglu.in"); Linear(block.GegluOut, name + ".geglu.out");
        }
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(63, parameters.Count); Assert.Equal(parameters.Count, expected.Count);
        Assert.All(parameters, parameter => Assert.True(expected.ContainsKey(parameter)));
        Tensor<double> Tensor(string name, params int[] shape) => new(shape, new Vector<double>(Values(reference.GetProperty(name))));
        var noisy = Tensor("noisy", 4, 4); var mask = Tensor("mask", 4, 1); var text = Tensor("text", 3, 6); var target = Tensor("target", 4, 4);
        using var tape = new GradientTape<double>();
        var output = model.Forward(noisy, mask, text, .37, example.GetProperty("drop").GetBoolean());
        Close(output, Values(example.GetProperty("prediction")), "complete velocity field");
        var error = engine.TensorMultiply(engine.TensorTile(mask, new[] { 1, 4 }), engine.TensorSubtract(output, target));
        var loss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(error, error), new[] { 0, 1 }, false), 1.0 / 8);
        Close(loss, new[] { example.GetProperty("loss").GetDouble() }, "masked velocity objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { noisy, text }).ToList());
        foreach (var parameter in parameters)
        {
            var wanted = expected[parameter];
            if (gradients.TryGetValue(parameter, out var gradient)) Close(gradient, wanted.Gradient, wanted.Name);
            else Assert.All(wanted.Gradient, value => Assert.Equal(0, value));
        }
        Close(gradients[noisy], Values(example.GetProperty("noisy_gradient")), "noisy-latent gradient");
        var textGradient = Values(example.GetProperty("text_gradient"));
        if (gradients.TryGetValue(text, out var actualTextGradient)) Close(actualTextGradient, textGradient, "text gradient");
        else Assert.All(textGradient, value => Assert.Equal(0, value));
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"Complete DiTTo {name} error {error:R}.");
    }
}
