using System.Reflection;
using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class E3UnetReferenceTests
{
    [Fact]
    public void AllTenResolutions_MatchTorchWaveformVarianceObjectiveAndEveryGradient()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "e3_unet_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement values) => values.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var entries = reference.GetProperty("parameters");
        var layers = new List<LayerBase<double>>();
        var unet = new E3WaveformUnet<double>(AiDotNetEngine.Current, layers, new E3TTSOptions
        {
            BlockChannels = Enumerable.Repeat(4, 10).ToArray(),
            AttentionHeads = new[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 },
            BertHiddenDim = 4, TimeEmbeddingDim = 8, ConditionDim = 4, SpeakerCount = 3, GroupNormGroups = 2
        }, new Random(17));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        unet.Load((name, shape) =>
        {
            var entry = entries.GetProperty(name);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        }, "");
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        static double[] Transpose(double[] values, int output, int input)
        {
            var result = new double[values.Length];
            for (int o = 0; o < output; o++) for (int i = 0; i < input; i++) result[i * output + o] = values[o * input + i];
            return result;
        }
        void Add(Tensor<double> tensor, string name, bool transpose = false)
        {
            var entry = entries.GetProperty(name);
            var weight = Values(entry.GetProperty("values"));
            var gradient = Values(entry.GetProperty("gradient"));
            if (transpose)
            {
                var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                weight = Transpose(weight, shape[0], shape[1]); gradient = Transpose(gradient, shape[0], shape[1]);
            }
            Close(tensor, weight, "import " + name, 1e-12);
            expected.Add(tensor, (name, gradient));
        }
        void Map(LayerBase<double> layer, string name)
        {
            if (layer is DenseLayer<double> dense)
            {
                Add(dense.GetWeights(), name + ".weight", true); Add(dense.GetBiases(), name + ".bias");
            }
            else if (layer is GroupNormalizationLayer<double> group)
            {
                Add(group.GetGammaTensor(), name + ".weight"); Add(group.GetBetaTensor(), name + ".bias");
            }
            else
            {
                var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
                Assert.InRange(parameters.Count, 1, 2);
                Add(parameters[0], name + ".weight");
                if (parameters.Count == 2) Add(parameters[1], name + ".bias");
            }
        }
        void Block(E3WaveBlock<double> block, string name)
        {
            for (int i = 0; i < block.FirstConvolutions.Count; i++) Map(block.FirstConvolutions[i], name + ".first." + i);
            // Reflection keeps this regression executable against the pre-repair production assembly.
            if (block.AdaptiveKernelCount > 0)
            {
                var property = typeof(E3WaveBlock<double>).GetProperty("SecondConvolutions");
                Assert.NotNull(property);
                var second = (IReadOnlyList<NormedConv1DLayer<double>>)property.GetValue(block)!;
                Assert.Equal(block.AdaptiveKernelCount, second.Count);
                for (int i = 0; i < second.Count; i++) Map(second[i], name + ".second." + i);
            }
            else Map(block.SecondConvolution, name + ".second");
            Map(block.FirstNorm, name + ".norm1"); Map(block.SecondNorm, name + ".norm2");
            if (block.Shortcut is not null) Map(block.Shortcut, name + ".shortcut");
            if (block.AdaptiveLogits is not null) Map(block.AdaptiveLogits, name + ".adaptive_logits");
            if (block.Film is not null) Map(block.Film, name + ".film");
            if (block.AttentionHeads == 0) return;
            Map(block.SelfNorm!, name + ".self.norm"); Map(block.SelfQuery!, name + ".self.q");
            Map(block.SelfKey!, name + ".self.k"); Map(block.SelfValue!, name + ".self.v"); Map(block.SelfOutput!, name + ".self.out");
            Map(block.CrossNorm!, name + ".cross.norm"); Map(block.CrossQuery!, name + ".cross.q");
            Map(block.CrossKey!, name + ".cross.k"); Map(block.CrossValue!, name + ".cross.v"); Map(block.CrossOutput!, name + ".cross.out");
        }
        Map(unet.TimeInput, "time.0"); Map(unet.TimeOutput, "time.2");
        Map(unet.SpeakerEmbedding, "speaker_embedding"); Map(unet.VarianceHead, "variance"); Map(unet.Output, "output");
        for (int i = 0; i < 10; i++)
        {
            Block(unet.DownBlocks[i], "down." + i); Block(unet.UpBlocks[i], "up." + i);
            for (int j = 0; j < unet.DownResiduals[i].Count; j++) Block(unet.DownResiduals[i][j], $"down.{i}.residual.{j}");
            for (int j = 0; j < unet.UpResiduals[i].Count; j++) Block(unet.UpResiduals[i][j], $"up.{i}.residual.{j}");
        }
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(885, parameters.Count); Assert.Equal(parameters.Count, expected.Count);
        Assert.All(parameters, parameter => Assert.True(expected.ContainsKey(parameter), "An owned tensor lacks a Torch gradient."));
        Tensor<double> Tensor(string name, params int[] shape) => new(shape, new Vector<double>(Values(reference.GetProperty(name))));
        var waveform = Tensor("waveform", 4097); var text = Tensor("text", 1, 3, 4);
        var noise = Tensor("noise", 4097); var weights = Tensor("weights", 4097);
        using var tape = new GradientTape<double>();
        var (epsilon, logVariance) = unet.Forward(waveform, text, .63, 1);
        Close(epsilon, Values(reference.GetProperty("epsilon")), "all-resolution epsilon");
        Close(logVariance, Values(reference.GetProperty("log_variance")), "timestep-only log variance");
        var engine = AiDotNetEngine.Current;
        var error = engine.TensorSubtract(epsilon, noise);
        var weighted = engine.TensorMultiply(weights, engine.TensorMultiply(error, error));
        var mse = engine.Reshape(engine.TensorMultiplyScalar(engine.ReduceSum(weighted, new[] { 0 }, true),
            1.0 / weights.ToArray().Sum()), new[] { 1, 1 });
        var loss = engine.TensorAdd(engine.TensorMultiply(engine.TensorExp(engine.TensorNegate(logVariance)), mse), logVariance);
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "complete weighted objective");
        var sources = parameters.Concat(new[] { waveform, text }).ToList();
        var gradients = tape.ComputeGradients(loss, sources: sources);
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.ContainsKey(parameter), "Missing gradient: " + expected[parameter].Name);
            Close(gradients[parameter], expected[parameter].Gradient, expected[parameter].Name);
        }
        Close(gradients[waveform], Values(reference.GetProperty("waveform_gradient")), "waveform gradient");
        Close(gradients[text], Values(reference.GetProperty("text_gradient")), "text-memory gradient");
    }

    private static void Close(Tensor<double> actual, double[] expected, string name, double tolerance = 1e-8)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < tolerance, $"E3 {name} maximum error {error:R}.");
    }
}
