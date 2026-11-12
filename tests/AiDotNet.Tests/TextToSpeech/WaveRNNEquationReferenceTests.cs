using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Independent dense-matrix PyTorch reference for the primary paper's recurrent equations.</summary>
public class WaveRNNEquationReferenceTests
{
    private sealed class EquationProbe : WaveRNN<double>
    {
        public EquationProbe() : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 2) { RandomSeed = 8 },
            new WaveRNNOptions
            {
                MelChannels = 2, UpsampleScales = [2], RnnDim = 4, SequenceSamples = 6,
                SampleRate = 4000, FftSize = 64, WindowSize = 64, MelMinFrequency = 0, MelMaxFrequency = 2000,
            }) { }
        public Tensor<double> Loss(Tensor<double> mel, Tensor<double> audio) => TrainingObjective(mel, audio, new Random(8));
    }

    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    private static void Close(Tensor<double> actual, double[] expected, string name, List<string> failures)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (!TtsPaperNumerics.IsFinite(actual[i]) || Math.Abs(actual[i] - expected[i]) > 1e-10)
            {
                failures.Add($"{name}[{i}]: {actual[i]:R} vs {expected[i]:R}");
                return;
            }
    }

    [Fact]
    public void CompleteCoarseFineLogits_Loss_AndEveryLearnedGradient_MatchPrimaryPaperEquations()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "wavernn_equations_reference.json")));
        var reference = document.RootElement;
        using var model = new EquationProbe();
        var layers = model.Layers.Cast<NormedConv1DLayer<double>>().ToArray();
        Assert.Equal(9, layers.Length);
        var parameters = new List<Tensor<double>>();
        for (int layer = 0; layer < layers.Length; layer++)
        {
            var tensors = layers[layer].GetTrainableParameters(); Assert.Equal(2, tensors.Count);
            var weight = Enumerable.Range(0, tensors[0].Length).Select(i => .15 * Math.Sin(.23 * (i + 1) + .41 * layer)).ToArray();
            var bias = Enumerable.Range(0, tensors[1].Length).Select(i => .03 * Math.Cos(.37 * (i + 1) + .19 * layer)).ToArray();
            layers[layer].LoadTorchWeights(weight, null, bias);
            parameters.AddRange(layers[layer].GetTrainableParameters());
        }
        Assert.Equal(1710, parameters.Sum(p => p.Length));
        var mel = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(Values(reference.GetProperty("mel"))));
        var audio = new Tensor<double>(new[] { 6 }, new Vector<double>(Values(reference.GetProperty("audio"))));
        var labels = model.Split(audio);
        Assert.Equal(reference.GetProperty("coarse").EnumerateArray().Select(x => x.GetInt32()), labels.Coarse);
        Assert.Equal(reference.GetProperty("fine").EnumerateArray().Select(x => x.GetInt32()), labels.Fine);
        var failures = new List<string>();
        using var tape = new GradientTape<double>();
        var logits = model.Logits(mel, audio);
        Close(logits.Coarse, Values(reference.GetProperty("coarseLogits")), "coarse logits", failures);
        Close(logits.Fine, Values(reference.GetProperty("fineLogits")), "fine logits", failures);
        var loss = model.Loss(mel, audio);
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "negative log-likelihood", failures);
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { mel }).ToList());
        var expectedGradients = reference.GetProperty("parameterGradients").EnumerateArray().ToArray();
        Assert.Equal(parameters.Count, expectedGradients.Length);
        for (int i = 0; i < parameters.Count; i++)
            Close(gradients[parameters[i]], Values(expectedGradients[i]), $"learned tensor {i} gradient", failures);
        Close(gradients[mel], Values(reference.GetProperty("melGradient")), "mel gradient", failures);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
