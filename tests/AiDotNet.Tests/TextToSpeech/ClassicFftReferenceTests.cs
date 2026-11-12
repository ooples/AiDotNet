using System.Reflection;
using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class ClassicFftReferenceTests
{
    [Fact]
    public void AdaSpeechConditionalNormalization_MatchesEquationOneAndAllSpeakerProjectionGradients()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "classic_fft_reference.json")));
        var reference = document.RootElement.GetProperty("conditional");
        double[] Values(string name) => reference.GetProperty(name).EnumerateArray().Select(value => value.GetDouble()).ToArray();
        var norm = new ConditionalLayerNormalizationLayer<double>(8, 6);
        TorchParameters.Linear(norm.ScaleProjection, 6, 8, Values("scale_weight"));
        TorchParameters.Linear(norm.BiasProjection, 6, 8, Values("shift_weight"));
        var input = new Tensor<double>(new[] { 2, 5, 8 }, new Vector<double>(Values("input")));
        var condition = new Tensor<double>(new[] { 2, 6 }, new Vector<double>(Values("condition")));
        var target = new Tensor<double>(new[] { 2, 5, 8 }, new Vector<double>(Values("target")));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { norm });
        Assert.Equal(2, parameters.Count);
        using var tape = new GradientTape<double>();
        var output = norm.Forward(input, condition);
        Close(output, Values("output"));
        var difference = AiDotNetEngine.Current.TensorSubtract(output, target);
        var loss = AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(difference, difference), new[] { 0, 1, 2 }, false);
        Assert.Equal(reference.GetProperty("loss").GetDouble(), loss[0], 9);
        var sources = parameters.Concat(new[] { input, condition }).ToList();
        var gradients = tape.ComputeGradients(loss, sources: sources);
        Close(gradients[input], Values("input_gradient")); Close(gradients[condition], Values("condition_gradient"));
        for (int i = 0; i < parameters.Count; i++)
        {
            var torchGradient = Values(i == 0 ? "scale_gradient" : "shift_gradient");
            var nativeGradient = new double[48];
            for (int row = 0; row < 8; row++) for (int column = 0; column < 6; column++) nativeGradient[column * 8 + row] = torchGradient[row * 6 + column];
            Close(gradients[parameters[i]], nativeGradient);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void PublishedFftKernels_MatchPyTorchOutputsLossAndEveryGradient(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "classic_fft_reference.json")));
        var reference = document.RootElement.GetProperty("cases")[index];
        double[] Values(JsonElement array) => array.EnumerateArray().Select(value => value.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => Values(reference.GetProperty("gradients").GetProperty(key));
        var kernels = reference.GetProperty("kernels").EnumerateArray().Select(value => value.GetInt32()).ToArray();
        var block = new FeedForwardTransformerBlock<double>(8, 2, 16, kernels[0], kernels[1], 0);
        TLayer Child<TLayer>(string field) => (TLayer)typeof(FeedForwardTransformerBlock<double>).GetField(field,
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(block)!;
        var sources = new List<Tensor<double>>(); var expectedGradients = new List<double[]>();
        void Copy(Tensor<double> tensor, double[] values)
        {
            Assert.Equal(values.Length, tensor.Length);
            for (int i = 0; i < values.Length; i++) tensor.SetFlat(i, values[i]);
            AiDotNetEngine.Current.InvalidatePersistentTensor(tensor);
        }
        static double[] Transpose(double[] values, int outputs, int inputs)
        {
            var result = new double[values.Length];
            for (int output = 0; output < outputs; output++)
                for (int input = 0; input < inputs; input++) result[input * outputs + output] = values[output * inputs + input];
            return result;
        }
        void Add(Tensor<double> tensor, double[] values, double[] gradients)
        { Copy(tensor, values); sources.Add(tensor); expectedGradients.Add(gradients); }
        var attention = Child<MultiHeadAttentionLayer<double>>("_attention");
        var qkv = Weight("attention.in_proj_weight"); var qkvGradient = Gradient("attention.in_proj_weight");
        Tensor<double>[] projections = [attention.GetQueryWeights(), attention.GetKeyWeights(), attention.GetValueWeights()];
        for (int i = 0; i < 3; i++)
            Add(projections[i], Transpose(qkv.Skip(i * 64).Take(64).ToArray(), 8, 8),
                Transpose(qkvGradient.Skip(i * 64).Take(64).ToArray(), 8, 8));
        Add(attention.GetOutputWeights(), Transpose(Weight("attention.out_proj.weight"), 8, 8),
            Transpose(Gradient("attention.out_proj.weight"), 8, 8));
        var outputBias = (Tensor<double>)typeof(MultiHeadAttentionLayer<double>).GetField("_outputBias",
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(attention)!;
        Add(outputBias, Weight("attention.out_proj.bias"), Gradient("attention.out_proj.bias"));
        foreach (var (field, name) in new[] { ("_attentionNorm", "norm1"), ("_convNorm", "norm2") })
        {
            var norm = Child<LayerNormalizationLayer<double>>(field);
            TorchParameters.LayerNorm(AiDotNetEngine.Current, norm, Weight(name + ".weight"), Weight(name + ".bias"));
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { norm });
            Assert.Equal(2, owned.Count);
            sources.AddRange(owned); expectedGradients.Add(Gradient(name + ".weight")); expectedGradients.Add(Gradient(name + ".bias"));
        }
        foreach (var (field, name) in new[] { ("_conv1", "conv1"), ("_conv2", "conv2") })
        {
            var conv = Child<Conv1DLayer<double>>(field);
            conv.SetParameters(new Vector<double>(Weight(name + ".weight").Concat(Weight(name + ".bias")).ToArray()));
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { conv });
            Assert.Equal(2, owned.Count);
            sources.AddRange(owned); expectedGradients.Add(Gradient(name + ".weight")); expectedGradients.Add(Gradient(name + ".bias"));
        }
        var input = new Tensor<double>(new[] { 2, 5, 8 }, new Vector<double>(Values(reference.GetProperty("input"))));
        var target = new Tensor<double>(new[] { 2, 5, 8 }, new Vector<double>(Values(reference.GetProperty("target"))));
        using var tape = new GradientTape<double>();
        var actual = block.Forward(input);
        Close(actual, Values(reference.GetProperty("output")));
        var error = AiDotNetEngine.Current.TensorSubtract(actual, target);
        var loss = AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(error, error), new[] { 0, 1, 2 }, false);
        Assert.Equal(reference.GetProperty("loss").GetDouble(), loss[0], 9);
        sources.Add(input); expectedGradients.Add(Values(reference.GetProperty("input_gradient")));
        var gradients = tape.ComputeGradients(loss, sources: sources);
        for (int i = 0; i < sources.Count; i++)
        {
            Assert.True(gradients.ContainsKey(sources[i]), $"Missing FFT gradient for source {i}.");
            Close(gradients[sources[i]], expectedGradients[i]);
        }
    }

    private static void Close(Tensor<double> actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"Maximum FFT output/gradient error {error:R}.");
    }
}
