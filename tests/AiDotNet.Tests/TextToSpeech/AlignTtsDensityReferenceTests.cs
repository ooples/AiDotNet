using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class AlignTtsDensityReferenceTests
{
    [Fact]
    public void DensityAndEnumeratedAlignmentNll_MatchAllInputAndNetworkGradients()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "aligntts_density_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement values) => values.EnumerateArray().Select(value => value.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        using var model = new AlignTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), new AlignTTSOptions
            { VocabSize = 32, EncoderDim = 8, HiddenDim = 8, NumHeads = 2, NumEncoderLayers = 1,
              NumDecoderLayers = 1, MelChannels = 4, FftFilterSize = 8, DurationPredictorDim = 8,
              MixDensityHiddenSize = 8, MixDensityHiddenLayers = 2, DropoutRate = 0 });
        // Inspect the actual private MDN rather than rebuilding its forward in the test.
        var layers = (List<LayerBase<double>>)typeof(AlignTTS<double>).GetField("_mixDensityNetwork",
            BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
        var parameters = new List<Tensor<double>>();
        var expectedGradients = new List<double[]>();
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is DenseLayer<double> dense) TorchParameters.Linear(AiDotNetEngine.Current, dense, 8, 8, Weight(i + ".weight"), Weight(i + ".bias"));
            else if (layers[i] is LayerNormalizationLayer<double> norm) TorchParameters.LayerNorm(AiDotNetEngine.Current, norm, Weight(i + ".weight"), Weight(i + ".bias"));
            else continue;
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<AiDotNet.Interfaces.ILayer<double>> { layers[i] });
            Assert.Equal(2, owned.Count);
            parameters.AddRange(owned);
            var weightGradient = Values(reference.GetProperty("gradients").GetProperty(i + ".weight"));
            if (layers[i] is DenseLayer<double>)
            {
                var transposed = new double[64];
                for (int output = 0; output < 8; output++)
                    for (int input = 0; input < 8; input++) transposed[input * 8 + output] = weightGradient[output * 8 + input];
                weightGradient = transposed;
            }
            expectedGradients.Add(weightGradient);
            expectedGradients.Add(Values(reference.GetProperty("gradients").GetProperty(i + ".bias")));
        }
        var hidden = new Tensor<double>(new[] { 3, 8 }, new Vector<double>(Values(reference.GetProperty("hidden"))));
        var mel = new Tensor<double>(new[] { 5, 4 }, new Vector<double>(Values(reference.GetProperty("mel"))));
        using var tape = new GradientTape<double>();
        var scores = (Tensor<double>)typeof(AlignTTS<double>).GetMethod("GaussianLogLikelihood",
            BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(model, new object[] { hidden, mel })!;
        Close(scores, Values(reference.GetProperty("scores")));
        var loss = model.AlignmentObjective(scores);
        Assert.Equal(reference.GetProperty("loss").GetDouble(), loss[0], 9);
        var sources = parameters.Concat(new[] { hidden, mel }).ToList();
        var gradients = tape.ComputeGradients(loss, sources: sources);
        for (int i = 0; i < parameters.Count; i++) Close(gradients[parameters[i]], expectedGradients[i]);
        Close(gradients[hidden], Values(reference.GetProperty("hidden_gradient")));
        Close(gradients[mel], Values(reference.GetProperty("mel_gradient")));
    }

    private static void Close(Tensor<double> actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"Maximum density/alignment error {error:R}.");
    }
}
