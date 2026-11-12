using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Independent PyTorch primitives for the paper's causal gated network and next-sample likelihood.</summary>
public class WaveNetEquationReferenceTests
{
    private sealed class Probe : WaveNet<double>
    {
        public Probe() : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 6) { RandomSeed = 8 },
            new WaveNetOptions
            {
                MelChannels = 2, UpsampleScales = [2, 1, 3], ResidualChannels = 3, GateChannels = 2, SkipChannels = 3,
                NumDilatedLayers = 3, DilationCycle = 2, KernelSize = 2, MuLawLevels = 16, SegmentSamples = 18,
                SampleRate = 4000, FftSize = 64, WindowSize = 64, MelMinFrequency = 0, MelMaxFrequency = 2000,
            }) { }
        public Tensor<double> Loss(Tensor<double> mel, Tensor<double> audio) => TrainingObjective(mel, audio, new Random(8));
    }

    private static double[] Values(JsonElement element) => element.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    private static void Close(Tensor<double> actual, double[] expected, string name, List<string> failures)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (!TtsPaperNumerics.IsFinite(actual.GetFlat(i)) || Math.Abs(actual.GetFlat(i) - expected[i]) > 1e-10)
            {
                failures.Add($"{name}[{i}]: {actual.GetFlat(i):R} vs {expected[i]:R}");
                return;
            }
    }

    [Fact]
    public void CompleteCausalNetwork_Logits_Nll_AndAllParameterAndMelGradients_MatchIndependentTorch()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "wavenet_equations_reference.json")));
        var reference = document.RootElement;
        using var model = new Probe();
        var layers = model.Layers.Cast<NormedConv1DLayer<double>>().ToArray();
        Assert.Equal(18, layers.Length);
        var parameters = new List<Tensor<double>>();
        var referenceLayers = reference.GetProperty("layers").EnumerateArray().ToArray();
        Assert.Equal(layers.Length, referenceLayers.Length);
        for (int i = 0; i < layers.Length; i++)
        {
            var item = referenceLayers[i]; var gain = item.GetProperty("gain");
            layers[i].LoadTorchWeights(Values(item.GetProperty("direction")),
                gain.ValueKind == JsonValueKind.Null ? null : Values(gain), Values(item.GetProperty("bias")));
            parameters.AddRange(layers[i].GetTrainableParameters());
        }
        Assert.Equal(447, reference.GetProperty("learned_scalars").GetInt32());
        Assert.Equal(447, parameters.Sum(p => p.Length));
        var mel = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(Values(reference.GetProperty("mel"))));
        var audio = new Tensor<double>(new[] { 18 }, new Vector<double>(Values(reference.GetProperty("audio"))));
        var labels = model.Quantize(audio);
        Assert.Equal(reference.GetProperty("labels").EnumerateArray().Select(x => x.GetInt32()), labels);
        var previous = new Tensor<double>(new[] { 1, 16, 18 });
        for (int t = 0; t < 18; t++) previous[0, t == 0 ? MuLaw.Encode(0, 16) : labels[t - 1], t] = 1;
        var network = (WaveNetNetwork<double>)typeof(WaveNet<double>).GetField("_network", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var failures = new List<string>();
        using var tape = new GradientTape<double>();
        Close(network.Forward(previous, network.Upsample(mel)), Values(reference.GetProperty("logits")), "logits", failures);
        Close(model.LogProbabilities(mel, audio), Values(reference.GetProperty("log_probabilities")), "log probabilities", failures);
        var loss = model.Loss(mel, audio);
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "negative log-likelihood", failures);
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { mel }).ToList());
        var expected = reference.GetProperty("parameter_gradients").EnumerateArray().ToArray();
        Assert.Equal(48, expected.Length);
        Assert.Equal(new[] { 36, 37, 38 }, Enumerable.Range(0, expected.Length)
            .Where(i => expected[i].ValueKind == JsonValueKind.Null).ToArray());
        Assert.Equal(parameters.Count, expected.Length);
        for (int i = 0; i < parameters.Count; i++)
        {
            // The last residual projection has no downstream consumer; its skip projection still contributes.
            if (expected[i].ValueKind == JsonValueKind.Null)
            {
                if (gradients.TryGetValue(parameters[i], out var unused))
                    Close(unused, new double[unused.Length], $"unused residual tensor {i}", failures);
            }
            else
            {
                var values = Values(expected[i]);
                Assert.True(values.Max(Math.Abs) > 1e-14, $"Oracle tensor {i} has no informative gradient.");
                Close(gradients[parameters[i]], values, $"parameter {i} gradient", failures);
            }
        }
        Close(gradients[mel], Values(reference.GetProperty("mel_gradient")), "mel gradient", failures);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
