using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
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
        public Tensor<double> StableLogs(Tensor<double> logits) => LogSoftmaxOverClasses(logits);
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


    [Fact]
    public void Stable_log_softmax_replays_changed_maxima_with_finite_reference_gradients()
    {
        using var model = new EquationProbe();
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<double>(new[] { 1, 3, 2 });
        var weights = new Tensor<double>(new[] { .2, -.4, .7, .1, -.3, .8 }, new[] { 1, 3, 2 });
        var parameters = new[] { logits };
        Tensor<double> Objective() => engine.ReduceSum(engine.TensorMultiply(model.StableLogs(logits), weights), null);
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 3; step++)
            {
                var values = Enumerable.Range(0, 6).Select(i => 1000 * Math.Sin(i + step * 1.7)).ToArray();
                for (int i = 0; i < values.Length; i++) logits.SetFlat(i, values[i]);
                logits.IncrementVersion();
                var expectedGradient = new double[6];
                double expectedLoss = 0;
                var scalarWeights = weights.ToArray();
                for (int t = 0; t < 2; t++)
                {
                    double max = Enumerable.Range(0, 3).Max(c => values[2 * c + t]);
                    double sum = Enumerable.Range(0, 3).Sum(c => Math.Exp(values[2 * c + t] - max));
                    double weightSum = Enumerable.Range(0, 3).Sum(c => scalarWeights[2 * c + t]);
                    for (int c = 0; c < 3; c++)
                    {
                        int index = 2 * c + t;
                        expectedLoss += scalarWeights[index] * (values[index] - max - Math.Log(sum));
                        expectedGradient[index] = scalarWeights[index] - weightSum * Math.Exp(values[index] - max) / sum;
                    }
                }
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                Assert.True(Math.Abs(plan.Step().ToArray()[0] - expectedLoss) < 1e-10);
                var failures = new List<string>();
                Close(plan.Gradients[0], expectedGradient, "stable log-softmax gradient", failures);
                Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
            }
        }
    }

    private static int[] ScalarCodes(IEnumerable<double> audio) => audio.Select(x =>
        (int)Math.Max(0, Math.Min(65535, Math.Round((Math.Max(-1, Math.Min(1, x)) + 1) * 32767.5)))).ToArray();

    [Fact]
    public void Quantized_float_targets_rebind_fresh_waveforms_without_foreign_producer_nodes()
    {
        var engine = new CpuEngine();
        var audio = new Tensor<float>(new[] { -.9f, -.5f, 0f, .2f, .7f, 1f }, new[] { 6 });
        var weight = new Tensor<float>(Enumerable.Repeat(0.1f, 12).ToArray(), new[] { 2, 6 });
        var parameters = new[] { weight };
        Tensor<float> Objective() => engine.ReduceSum(engine.TensorMultiply(
            WaveRNN<float>.QuantizedBytes(engine, audio), weight), null);
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 4; step++)
            {
                audio = new Tensor<float>(Enumerable.Range(0, 6)
                    .Select(i => (float)(1.2 * Math.Sin(i + step * .43))).ToArray(), new[] { 6 });
                var codes = ScalarCodes(audio.ToArray().Select(x => (double)x));
                var expected = codes.Select(code => (float)(code >> 8))
                    .Concat(codes.Select(code => (float)(code & 255))).ToArray();
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                _ = plan.Step();
                Assert.Equal(expected, plan.Gradients[0].ToArray());
                Assert.All(weight.ToArray(), value => Assert.Equal(.1f, value));
            }
        }
    }

    [Fact]
    public void Actual_model_replays_current_teacher_forcing_targets_and_every_learned_gradient()
    {
        using var model = new EquationProbe();
        var mel = new Tensor<double>(Enumerable.Range(0, 6).Select(i => .1 * Math.Sin(i)).ToArray(), new[] { 1, 2, 3 });
        var audio = new Tensor<double>(new[] { -1.2, -.5, 0, .25, .8, 1.2 }, new[] { 6 });
        _ = model.Loss(mel, audio);
        var parameters = model.Layers.Cast<NormedConv1DLayer<double>>()
            .SelectMany(layer => layer.GetTrainableParameters()).ToArray();
        Assert.Equal(18, parameters.Length);
        var manifest = model.CollectModelTrainableTensors();
        Assert.Equal(manifest.Count, parameters.Length);
        Assert.All(manifest, parameter => Assert.True(parameters.Any(candidate => ReferenceEquals(parameter, candidate))));
        Assert.Equal(model.TrainableParameterCount, parameters.Sum(parameter => (long)parameter.Length));
        var before = parameters.Select(parameter => parameter.ToArray()).ToArray();
        Tensor<double> Objective() => model.Loss(mel, audio);
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 4; step++)
            {
                mel = new Tensor<double>(Enumerable.Range(0, 6)
                    .Select(i => .1 * Math.Sin(i + step * .31)).ToArray(), new[] { 1, 2, 3 });
                audio = new Tensor<double>(Enumerable.Range(0, 6)
                    .Select(i => 1.15 * Math.Sin(i * .7 + step * .29)).ToArray(), new[] { 6 });
                var codes = ScalarCodes(audio.ToArray());
                var labels = model.Split(audio);
                Assert.Equal(codes.Select(code => code >> 8), labels.Coarse);
                Assert.Equal(codes.Select(code => code & 255), labels.Fine);
                var logits = model.Logits(mel, audio);
                double CrossEntropy(Tensor<double> values, Func<int, int> label)
                {
                    var flat = values.ToArray();
                    double loss = 0;
                    for (int t = 0; t < 6; t++)
                    {
                        double max = Enumerable.Range(0, 256).Max(c => flat[c * 6 + t]);
                        double sum = Enumerable.Range(0, 256).Sum(c => Math.Exp(flat[c * 6 + t] - max));
                        loss += Math.Log(sum) + max - flat[label(codes[t]) * 6 + t];
                    }
                    return loss / 6;
                }
                double expectedLoss = CrossEntropy(logits.Coarse, code => code >> 8)
                    + CrossEntropy(logits.Fine, code => code & 255);
                double[][] expectedGradients;
                using (var tape = new GradientTape<double>())
                {
                    var eagerLoss = Objective();
                    Assert.True(Math.Abs(eagerLoss.ToArray()[0] - expectedLoss) < 1e-10);
                    var gradients = tape.ComputeGradients(eagerLoss, parameters);
                    expectedGradients = parameters.Select(parameter => gradients[parameter].ToArray()).ToArray();
                }
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                Assert.True(Math.Abs(plan.Step().ToArray()[0] - expectedLoss) < 1e-10);
                var failures = new List<string>();
                for (int parameter = 0; parameter < parameters.Length; parameter++)
                {
                    Close(plan.Gradients[parameter], expectedGradients[parameter],
                        $"replayed learned tensor {parameter}", failures);
                    Assert.Equal(before[parameter], parameters[parameter].ToArray());
                }
                Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
            }
        }
    }
}
