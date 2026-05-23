// SenseVoice training-step profile / regression harness.
//
// Decomposes a SenseVoice training step into its measurable phases —
// constructor + InitializeLayers, lazy-init forward, warm forward,
// warm Train, ForwardForTraining inside the tape — and emits each
// timing via ITestOutputHelper so the breakdown is visible in CI
// logs. Used together with the Tensors-side perf work (#1421) to
// keep the backward+optimizer half of the step within the budget
// Train_PhaseBudgets_AreEnforced asserts on, so a future regression
// (e.g. a tape-recording inefficiency that doubles backward time)
// fails this test rather than silently slowing down every test class
// that exercises SenseVoice.Train.

using System.Collections.Generic;
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

    // Per-phase budgets, calibrated against the timings observed on
    // the post-Tensors-#1421 fast path with CPU engine. Generous
    // (2.5×–4× of the measured median) so CI noise on slower runners
    // doesn't false-fail the regression check, but tight enough that
    // a real backward-pass regression (e.g. tape recording 2× more
    // ops, optimizer step missing a vectorized path) trips the
    // assertion. Values in milliseconds.
    private const double CtorBudgetMs = 30_000;
    private const double WarmPredictBudgetMs = 1_500;
    private const double TrainStepBudgetMs = 12_000;
    private const double ForwardForTrainingBudgetMs = 2_000;

    [Fact(Timeout = 600000)]
    public void Profile_StepBreakdown()
    {
        AiDotNetEngine.ResetToCpu();

        // SenseVoice-Small paper-faithful defaults (Du et al. 2024)
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32, outputSize: 25000);
        var swCtor = Stopwatch.StartNew();
        using var model = new SenseVoice<float>(arch);
        swCtor.Stop();
        double ctorMs = swCtor.Elapsed.TotalMilliseconds;
        _output.WriteLine($"Ctor + InitializeLayers: {ctorMs:F1} ms");
        _output.WriteLine($"Layers.Count = {model.Layers.Count}, ParameterCount = {model.ParameterCount:N0}");

        Assert.True(ctorMs <= CtorBudgetMs,
            $"SenseVoice ctor+InitializeLayers took {ctorMs:F0} ms — over the {CtorBudgetMs:F0} ms budget. "
            + "Suspect a regression in layer init (e.g. eager weight materialization on a layer that "
            + "should be lazy, or a missing cache hit on shared embeddings).");
        Assert.True(model.Layers.Count > 0, "SenseVoice initialized with zero layers.");
        Assert.True(model.ParameterCount > 0, "SenseVoice has zero parameters after init.");

        var rng = RandomHelper.CreateSeededRandom(0);
        var input = new Tensor<float>([1, 64, 32]);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        var swP1 = Stopwatch.StartNew();
        var pred1 = model.Predict(input);
        swP1.Stop();
        _output.WriteLine($"Predict #1 (lazy init): {swP1.Elapsed.TotalMilliseconds:F1} ms, output shape [{string.Join(",", pred1.Shape)}]");

        var swP2 = Stopwatch.StartNew();
        var pred2 = model.Predict(input);
        swP2.Stop();
        double warmPredictMs = swP2.Elapsed.TotalMilliseconds;
        _output.WriteLine($"Predict #2 (warm):      {warmPredictMs:F1} ms");

        Assert.True(warmPredictMs <= WarmPredictBudgetMs,
            $"SenseVoice warm Predict took {warmPredictMs:F0} ms — over the {WarmPredictBudgetMs:F0} ms budget. "
            + "Suspect a regression in the forward inference path (engine routing, missing fast path).");
        Assert.False(double.IsNaN(pred2[0]) || double.IsInfinity(pred2[0]),
            "SenseVoice Predict produced NaN/Inf — forward path is numerically broken.");

        // Make a target matching the predict output shape so Train can compute a loss.
        var predShape = new int[pred1.Shape.Length];
        for (int i = 0; i < predShape.Length; i++) predShape[i] = pred1.Shape[i];
        var target = new Tensor<float>(predShape);
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);

        var swT1 = Stopwatch.StartNew();
        model.Train(input, target);
        swT1.Stop();
        _output.WriteLine($"Train #1 (warm-up): {swT1.Elapsed.TotalMilliseconds:F1} ms");

        // Measure 3 warm Train iterations and take the median as the
        // representative per-step cost — first Train tends to pay
        // additional one-shot costs (tape pre-allocation, optimizer
        // moment-vector init).
        var trainTimings = new List<double>();
        for (int i = 0; i < 3; i++)
        {
            var swTn = Stopwatch.StartNew();
            model.Train(input, target);
            swTn.Stop();
            double ms = swTn.Elapsed.TotalMilliseconds;
            trainTimings.Add(ms);
            _output.WriteLine($"Train #{i+2}:            {ms:F1} ms");
        }
        trainTimings.Sort();
        double medianTrainMs = trainTimings[trainTimings.Count / 2];
        Assert.True(medianTrainMs <= TrainStepBudgetMs,
            $"SenseVoice median warm Train step took {medianTrainMs:F0} ms — over the {TrainStepBudgetMs:F0} ms "
            + "budget. Backward pass + optimizer step is regressing; check the Tensors-side tape recording "
            + "and Adam fast paths.");

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

        Assert.True(forwardTrainingAvgMs <= ForwardForTrainingBudgetMs,
            $"SenseVoice ForwardForTraining took {forwardTrainingAvgMs:F0} ms — over the "
            + $"{ForwardForTrainingBudgetMs:F0} ms budget. The tape-recording forward path is regressing; "
            + "check whether a layer's training-mode forward is doing extra work that the inference-mode "
            + $"forward (~{forwardAvgMs:F0} ms above) avoids.");

        // Derived metric: backward + optimizer = train_step − forward.
        // Replaces the previous hardcoded 4500 ms estimate — now we use
        // the actual measured Train cost from above so the breakdown
        // stays accurate as the budgets evolve.
        double backwardPlusOptMs = medianTrainMs - forwardTrainingAvgMs;
        _output.WriteLine($"=> Backward + optimizer ≈ Train ({medianTrainMs:F0} ms) "
                          + $"− ForwardForTraining ({forwardTrainingAvgMs:F0} ms) "
                          + $"= {backwardPlusOptMs:F0} ms");

        // Sanity: Train must dominate forward; if not, the tape isn't
        // engaging (e.g. ForwardForTraining is falling through to
        // inference path, or Train is silently no-op'ing).
        Assert.True(medianTrainMs > forwardTrainingAvgMs * 0.9,
            $"SenseVoice Train ({medianTrainMs:F0} ms) is unexpectedly close to or below ForwardForTraining "
            + $"({forwardTrainingAvgMs:F0} ms) — the backward + optimizer portion of Train is missing. "
            + "Likely a regression where Train silently became inference-mode no-op (e.g. tape registry "
            + "leaked from a prior test, or training-mode flag not propagating).");
    }
}
