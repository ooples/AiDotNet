// Throwaway profile harness for SenseVoice training-step breakdown.
// Asserts nothing — just emits timings via _output so the perf bottleneck
// surfaces in the test log. Delete once #1421 follow-up performance work
// lands.

using System.Diagnostics;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Performance;

public class SenseVoiceTrainStepProfile
{
    private readonly ITestOutputHelper _output;
    public SenseVoiceTrainStepProfile(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 600000)]
    public async System.Threading.Tasks.Task Profile_StepBreakdown()
    {
        await System.Threading.Tasks.Task.Yield();
        AiDotNetEngine.ResetToCpu();

        // SenseVoice-Small paper-faithful defaults (Du et al. 2024)
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32, outputSize: 25000);
        var swCtor = Stopwatch.StartNew();
        using var model = new SenseVoice<float>(arch);
        swCtor.Stop();
        _output.WriteLine($"Ctor + InitializeLayers: {swCtor.Elapsed.TotalMilliseconds:F1} ms");
        _output.WriteLine($"Layers.Count = {model.Layers.Count}, ParameterCount = {model.ParameterCount:N0}");

        var rng = RandomHelper.CreateSeededRandom(0);
        var input = new Tensor<float>([1, 64, 32]);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        var swP1 = Stopwatch.StartNew();
        var pred1 = model.Predict(input);
        swP1.Stop();
        _output.WriteLine($"Predict #1 (lazy init): {swP1.Elapsed.TotalMilliseconds:F1} ms, output shape [{string.Join(",", pred1.Shape)}]");

        var swP2 = Stopwatch.StartNew();
        _ = model.Predict(input);
        swP2.Stop();
        _output.WriteLine($"Predict #2 (warm):      {swP2.Elapsed.TotalMilliseconds:F1} ms");

        // Make a target matching the predict output shape so Train can compute a loss.
        var predShape = new int[pred1.Shape.Length];
        for (int i = 0; i < predShape.Length; i++) predShape[i] = pred1.Shape[i];
        var target = new Tensor<float>(predShape);
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);

        var swT1 = Stopwatch.StartNew();
        model.Train(input, target);
        swT1.Stop();
        _output.WriteLine($"Train #1 (warm-up): {swT1.Elapsed.TotalMilliseconds:F1} ms");

        for (int i = 0; i < 3; i++)
        {
            var swTn = Stopwatch.StartNew();
            model.Train(input, target);
            swTn.Stop();
            _output.WriteLine($"Train #{i+2}:            {swTn.Elapsed.TotalMilliseconds:F1} ms");
        }

        // Time the forward-only path (no tape) over many iters to confirm
        // forward is the small fraction.
        const int N = 5;
        long forwardTicks = 0;
        for (int i = 0; i < N; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = model.Predict(input);
            sw.Stop();
            forwardTicks += sw.ElapsedTicks;
        }
        double forwardAvgMs = (forwardTicks * 1000.0 / Stopwatch.Frequency) / N;
        _output.WriteLine($"Predict avg ({N} iters): {forwardAvgMs:F1} ms");

        // Forward through tape (training mode) to see how much of Train's
        // time is the forward portion vs backward + optimizer.
        long forwardTrainingTicks = 0;
        for (int i = 0; i < N; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = model.ForwardForTraining(input);
            sw.Stop();
            forwardTrainingTicks += sw.ElapsedTicks;
        }
        double forwardTrainingAvgMs = (forwardTrainingTicks * 1000.0 / Stopwatch.Frequency) / N;
        _output.WriteLine($"ForwardForTraining avg ({N} iters): {forwardTrainingAvgMs:F1} ms (= forward inside Train)");
        _output.WriteLine($"=> Backward + optimizer = Train ({4500:F0} ms approx) - ForwardForTraining ({forwardTrainingAvgMs:F1} ms) ≈ {4500 - forwardTrainingAvgMs:F0} ms");
    }
}
