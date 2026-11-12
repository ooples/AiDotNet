using System.Collections;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class SeedDiffusionReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void CompleteDenoiserAtMultipleNoiseLevelsAndDdim_MatchTorchAndEveryGradient(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "seed_diffusion_reference.json")));
        var reference = document.RootElement; var example = reference.GetProperty("cases")[index];
        double[] Values(JsonElement values) => values.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => Values(example.GetProperty("gradients").GetProperty(key));
        var options = new SeedTTSOptions { AcousticDimension = 4, CodebookSize = 8, SpeakerEmbeddingDim = 3,
            NumHeads = 2, DiffusionLayers = 1, DiffusionSteps = 4, MaxCodecFrames = 8 };
        void Option(string name, object value)
        {
            var property = typeof(SeedTTSOptions).GetProperty(name);
            Assert.NotNull(property); property.SetValue(options, value);
        }
        Option("DiffusionTrainingSteps", 8); Option("DiffusionBetaStart", .01); Option("DiffusionBetaEnd", .1);
        var layers = new List<LayerBase<double>>();
        var diffusion = new SeedTTSTokenDiffusion<double>(AiDotNetEngine.Current, layers, options);
        foreach (var layer in layers) layer.SetTrainingMode(false);
        var tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 2.0, 4.0, 6.0 }));
        Tensor<double> Tensor(string key, params int[] shape) => new(shape, new Vector<double>(Values(reference.GetProperty(key))));
        var target = Tensor("target", 3, 4); var noise = Tensor("noise", 3, 4); var timbre = Tensor("timbre", 3);
        object Invoke(string name, params object[] arguments) => typeof(SeedTTSTokenDiffusion<double>).GetMethods(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic).Single(method => method.Name == name &&
                method.GetParameters().Length == arguments.Length).Invoke(diffusion, arguments)!;
        Invoke("Predict", target, tokens, timbre, .5); // Materialize before assigning independent weights.
        static T Field<T>(object owner, string name) => (T)owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;
        static T Property<T>(object owner, string name) => (T)owner.GetType().GetProperty(name,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.GetValue(owner)!;
        static double[] Transpose(double[] values, int output, int input)
        {
            var result = new double[values.Length];
            for (int o = 0; o < output; o++) for (int i = 0; i < input; i++) result[i * output + o] = values[o * input + i];
            return result;
        }
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        void Copy(Tensor<double> tensor, string name, double[]? weight = null, double[]? gradient = null)
        {
            weight ??= Weight(name); gradient ??= Gradient(name);
            Assert.Equal(tensor.Length, weight.Length);
            for (int i = 0; i < weight.Length; i++) tensor.SetFlat(i, weight[i]);
            AiDotNetEngine.Current.InvalidatePersistentTensor(tensor);
            expected.Add(tensor, (name, gradient));
        }
        void Map(LayerBase<double> layer, string name)
        {
            if (layer is DenseLayer<double> dense)
            {
                var bias = Weight(name + ".bias"); int outputs = bias.Length, inputs = Weight(name + ".weight").Length / outputs;
                Copy(dense.GetWeights(), name + ".weight", Transpose(Weight(name + ".weight"), outputs, inputs), Transpose(Gradient(name + ".weight"), outputs, inputs));
                Copy(dense.GetBiases(), name + ".bias");
            }
            else if (layer is FullyConnectedLayer<double> fullyConnected)
            {
                Copy(fullyConnected.GetWeights()!, name + ".weight"); Copy(fullyConnected.GetBiases()!, name + ".bias");
            }
            else
            {
                var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
                Assert.InRange(owned.Count, 1, 2); Copy(owned[0], name + ".weight");
                if (owned.Count == 2) Copy(owned[1], name + ".bias");
            }
        }
        Map(Field<TiedEmbeddingLayer<double>>(diffusion, "_tokens"), "tokens");
        Map(Field<FullyConnectedLayer<double>>(diffusion, "_noisyInput"), "noisy");
        Map(Field<FullyConnectedLayer<double>>(diffusion, "_timbre"), "timbre");
        Map(Field<FullyConnectedLayer<double>>(diffusion, "_output"), "output");
        int blockIndex = 0;
        foreach (object block in Field<IEnumerable>(diffusion, "_blocks"))
        {
            string name = "blocks." + blockIndex++;
            var attention = Property<MultiHeadAttentionLayer<double>>(block, "Attention");
            Tensor<double>[] qkv = [attention.GetQueryWeights(), attention.GetKeyWeights(), attention.GetValueWeights()];
            var weight = Weight(name + ".attention.in_proj_weight"); var gradient = Gradient(name + ".attention.in_proj_weight");
            for (int i = 0; i < 3; i++) Copy(qkv[i], name + ".qkv." + i,
                Transpose(weight.Skip(i * 16).Take(16).ToArray(), 4, 4), Transpose(gradient.Skip(i * 16).Take(16).ToArray(), 4, 4));
            Copy(attention.GetOutputWeights(), name + ".attention.out_proj.weight", Transpose(Weight(name + ".attention.out_proj.weight"), 4, 4),
                Transpose(Gradient(name + ".attention.out_proj.weight"), 4, 4));
            Copy(Field<Tensor<double>>(attention, "_outputBias"), name + ".attention.out_proj.bias");
            Map(Property<LayerNormalizationLayer<double>>(block, "Norm1"), name + ".norm1");
            Map(Property<LayerNormalizationLayer<double>>(block, "Norm2"), name + ".norm2");
            Map(Property<DenseLayer<double>>(block, "FeedForwardIn"), name + ".ff1");
            Map(Property<DenseLayer<double>>(block, "FeedForwardOut"), name + ".ff2");
        }
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(20, parameters.Count); Assert.Equal(parameters.Count, expected.Count);
        Assert.All(parameters, parameter => Assert.True(expected.ContainsKey(parameter)));
        var alpha = Values(reference.GetProperty("alpha"));
        for (int i = 0; i < alpha.Length; i++) Assert.True(Math.Abs((double)Invoke("TrainingAlphaBar", i) - alpha[i]) < 1e-14);
        int timestep = example.GetProperty("timestep").GetInt32();
        var noisy = (Tensor<double>)Invoke("Corrupt", target, noise, timestep);
        Close(noisy, Values(example.GetProperty("noisy")), "VP-corrupted features");
        Close((Tensor<double>)Invoke("Predict", noisy, tokens, timbre, timestep / 8.0), Values(example.GetProperty("predicted")), "complete clean prediction");
        if (index == 0)
        {
            var state = noise;
            for (int step = 4; step > 0; step--)
            {
                var clean = (Tensor<double>)Invoke("Predict", state, tokens, timbre, step / 4.0);
                state = (Tensor<double>)Invoke("DdimStep", state, clean, step * 2, (step - 1) * 2);
                Close(state, Values(reference.GetProperty("trajectory")[4 - step]), "DDIM step " + step);
            }
            Close((Tensor<double>)Invoke("Refine", tokens, timbre, noise), Values(reference.GetProperty("sampled")), "actual complete DDIM sampler");
        }
        using var tape = new GradientTape<double>();
        var draws = new OracleDraws(timestep, noise.ToArray());
        var loss = (Tensor<double>)Invoke("DenoisingLoss", tokens, target, timbre, draws);
        Assert.Empty(draws.Uniforms);
        Close(loss, new[] { example.GetProperty("loss").GetDouble() }, "actual sampled training objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { target, noise, timbre }).ToList());
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.ContainsKey(parameter), "Missing denoiser gradient " + expected[parameter].Name);
            Close(gradients[parameter], expected[parameter].Gradient, expected[parameter].Name);
        }
        Close(gradients[target], Values(example.GetProperty("target_gradient")), "target gradient");
        Close(gradients[timbre], Values(example.GetProperty("timbre_gradient")), "timbre gradient");
        // Independently supplied noise remains differentiable in the explicit-timestep objective.
        using var noiseTape = new GradientTape<double>();
        var explicitLoss = (Tensor<double>)Invoke("DenoisingLossAtTimestep", tokens, target, timbre, timestep, noise);
        var noiseGradient = noiseTape.ComputeGradients(explicitLoss, sources: new[] { noise });
        Close(noiseGradient[noise], Values(example.GetProperty("noise_gradient")), "noise gradient");
    }

    private sealed class OracleDraws : Random
    {
        private readonly int _timestep;
        internal Queue<double> Uniforms { get; } = new();
        internal OracleDraws(int timestep, double[] gaussian)
        {
            _timestep = timestep;
            foreach (double value in gaussian)
            {
                Uniforms.Enqueue(1 - Math.Exp(-value * value / 2)); Uniforms.Enqueue(value < 0 ? .5 : 0);
            }
        }
        public override int Next(int minValue, int maxValue)
        {
            Assert.InRange(_timestep, minValue, maxValue - 1); return _timestep;
        }
        public override double NextDouble() => Uniforms.Dequeue();
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"Seed diffusion {name} error {error:R}.");
    }
}
