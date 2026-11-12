using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceReferenceEncoderTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OfficialSixConvolutionsAndResetAfterGru_MatchEveryParameterAndInputGradient(bool normalize)
    {
        string directory = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData");
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "openvoice_reference_encoder.json")));
        var item = metadata.RootElement.GetProperty("cases").EnumerateArray()
            .Single(x => x.GetProperty("layer_normalization").GetBoolean() == normalize);
        string label = item.GetProperty("label").GetString()!;
        using var stream = File.OpenRead(Path.Combine(directory, "openvoice_reference_encoder.safetensors"));
        using var reader = new BinaryReader(stream);
        long headerLength = reader.ReadInt64();
        using var header = JsonDocument.Parse(reader.ReadBytes(checked((int)headerLength)));
        double[] Read(string name, int[]? shape = null)
        {
            var entry = header.RootElement.GetProperty(label + "." + name);
            Assert.Equal("F64", entry.GetProperty("dtype").GetString());
            if (shape is not null)
                Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            var offsets = entry.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            stream.Position = 8 + headerLength + offsets[0];
            var values = new double[checked((int)((offsets[1] - offsets[0]) / 8))];
            for (int i = 0; i < values.Length; i++) values[i] = reader.ReadDouble();
            return values;
        }
        var engine = AiDotNetEngine.Current;
        var model = new OpenVoiceReferenceEncoder<double>(engine, 17, 7, normalize);
        foreach (var layer in model.Layers) layer.SetTrainingMode(false);
        var imported = new HashSet<string>();
        model.LoadTorch((name, shape) =>
        {
            Assert.True(imported.Add(name), "Duplicate imported parameter: " + name);
            return Read("weight." + name, shape);
        }, "");
        Assert.Equal(item.GetProperty("parameters").GetArrayLength(), imported.Count);
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        void Map(Tensor<double> parameter, string name, bool transpose = false)
        {
            double[] weights = Read("weight." + name), gradient = Read("gradient." + name);
            if (transpose)
            {
                int inputs = parameter.Shape[0], outputs = parameter.Shape[1];
                double[] Transpose(double[] source)
                {
                    var result = new double[source.Length];
                    for (int i = 0; i < inputs; i++) for (int o = 0; o < outputs; o++)
                        result[i * outputs + o] = source[o * inputs + i];
                    return result;
                }
                weights = Transpose(weights); gradient = Transpose(gradient);
            }
            Close(parameter, weights, "import " + name, 1e-12);
            expected.Add(parameter, (name, gradient));
        }
        int convolution = 0, denseIndex = 0;
        foreach (var layer in model.Layers)
        {
            if (layer is LayerNormalizationLayer<double> norm)
            {
                Map(norm.GetGammaTensor(), "layernorm.weight"); Map(norm.GetBetaTensor(), "layernorm.bias");
            }
            else if (layer is DenseLayer<double> dense)
            {
                string[] weights = { "gru.weight_ih_l0", "gru.weight_hh_l0", "proj.weight" };
                string[] biases = { "gru.bias_ih_l0", "gru.bias_hh_l0", "proj.bias" };
                Map(dense.GetWeights(), weights[denseIndex], true);
                Map(dense.GetBiases(), biases[denseIndex++]);
            }
            else
            {
                var p = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
                Assert.Equal(3, p.Count);
                Map(p[0], $"convs.{convolution}.weight_v");
                Map(p[1], $"convs.{convolution}.weight_g");
                Map(p[2], $"convs.{convolution++}.bias");
            }
        }
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(normalize ? 26 : 24, parameters.Count);
        Assert.Equal(parameters.Count, expected.Count);
        var input = new Tensor<double>(new[] { 2, 129, 17 }, new Vector<double>(Read("input")));
        var objectiveWeights = new Tensor<double>(new[] { 2, 7 }, new Vector<double>(Read("objective_weights")));
        using var tape = new GradientTape<double>();
        var output = model.Forward(input);
        Close(output, Read("output"), "speaker vector");
        var loss = engine.ReduceSum(engine.TensorMultiply(output, objectiveWeights), new[] { 0, 1 }, true);
        Close(loss, new[] { item.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { input }).ToList());
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.ContainsKey(parameter), "Missing gradient: " + expected[parameter].Name);
            Close(gradients[parameter], expected[parameter].Gradient, expected[parameter].Name);
        }
        Close(gradients[input], Read("input_gradient"), "spectrogram gradient");
    }

    private static void Close(Tensor<double> actual, double[] expected, string name, double tolerance = 1e-8)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < tolerance, $"OpenVoice {name} maximum error {error:R}.");
    }
}
