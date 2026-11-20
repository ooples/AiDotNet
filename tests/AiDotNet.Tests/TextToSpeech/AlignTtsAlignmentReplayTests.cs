using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class AlignTtsAlignmentReplayTests
{
    private static int PrefixLength(double[] scores)
    {
        // Independently enumerate every legal two-token path; each token owns a nonempty frame interval.
        double best = double.NegativeInfinity;
        int selected = 0;
        for (int split = 1; split < 4; split++)
        {
            double value = 0;
            for (int frame = 0; frame < 4; frame++) value += scores[(frame < split ? 0 : 4) + frame];
            if (value > best) { best = value; selected = split; }
        }
        return selected;
    }

    [Fact]
    public void Compiled_alignment_refreshes_paths_and_duration_targets_without_differentiating_the_hard_search()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var cases = new[]
            {
                new[] { 8.0, 7, -8, -9, -9, -8, 7, 8 },
                new[] { 9.0, -3, -5, -8, -8, 3, 5, 9 },
                new[] { 9.0, 6, 4, -9, -9, -4, -6, 8 },
                new[] { 8.0, 7, -8, -9, -9, -8, 7, 8 }
            };
            Assert.Equal(new[] { 2, 1, 3, 2 }, cases.Select(PrefixLength).ToArray());
            var scores = new Tensor<double>(cases[0], new[] { 2, 4 });
            var hidden = new Tensor<double>(new[] { 2.0, 3, 5, 7 }, new[] { 2, 2 });
            var predicted = new Tensor<double>(new[] { 0.1, -0.2 }, new[] { 2 });
            var parameters = new[] { hidden, scores, predicted };
            var frameWeights = new Tensor<double>(new[] { 1.0, 2, 3, 4, 5, 6, 7, 8 }, new[] { 4, 2 });
            Tensor<double> Objective()
            {
                var path = TtsModelBase<double>.MonotonicAlignmentPath(engine, scores);
                var expanded = engine.TensorMatMul(engine.TensorTranspose(path), hidden);
                var target = engine.TensorLog1p(engine.ReduceSum(path, new[] { 1 }, false));
                var durationError = engine.TensorSubtract(predicted, target);
                return engine.TensorAdd(engine.TensorAdd(
                    engine.ReduceSum(engine.TensorMultiply(expanded, frameWeights), null),
                    engine.ReduceSum(engine.TensorMultiply(durationError, durationError), null)),
                    engine.TensorMultiplyScalar(engine.ReduceSum(scores, null), 0.25));
            }
            using var recording = ClosureObjectiveGuard<double>.BeginRecording();
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
                plan = scope.CompileTraining(parameters, Objective());
            using (plan)
            {
                var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
                Assert.True(guard is not null, reason);
                guard!.AttachTo(plan);
                for (int step = 0; step < cases.Length; step++)
                {
                    for (int i = 0; i < scores.Length; i++) scores.SetFlat(i, cases[step][i]);
                    scores.IncrementVersion();
                    predicted.SetFlat(0, 0.1 + step * 0.2);
                    predicted.SetFlat(1, -0.2 + step * 0.05);
                    predicted.IncrementVersion();
                    Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                    int prefix = PrefixLength(cases[step]);
                    var hiddenGradient = new double[4];
                    double expectedLoss = 0.25 * cases[step].Sum();
                    for (int frame = 0; frame < 4; frame++)
                        for (int feature = 0; feature < 2; feature++)
                        {
                            int token = frame < prefix ? 0 : 1;
                            hiddenGradient[token * 2 + feature] += frameWeights[frame, feature];
                            expectedLoss += hidden[token, feature] * frameWeights[frame, feature];
                        }
                    var durationGradient = new double[2];
                    for (int token = 0; token < 2; token++)
                    {
                        int duration = token == 0 ? prefix : 4 - prefix;
                        double error = predicted[token] - Math.Log(1 + duration);
                        expectedLoss += error * error;
                        durationGradient[token] = 2 * error;
                    }
                    Assert.Equal(expectedLoss, plan.Step().ToArray()[0], 10);
                    Assert.Equal(hiddenGradient, plan.Gradients[0].ToArray());
                    // Only the separate continuous score term differentiates through scores, as in joint training.
                    Assert.All(plan.Gradients[1].ToArray(), value => Assert.Equal(0.25, value));
                    var actualDurationGradient = plan.Gradients[2].ToArray();
                    for (int token = 0; token < 2; token++) Assert.Equal(durationGradient[token], actualDurationGradient[token], 10);
                    Assert.Equal(new[] { 2.0, 3, 5, 7 }, hidden.ToArray());
                    Assert.Equal(cases[step], scores.ToArray());
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(2049)]
    [InlineData(4097)]
    public void Half_duration_target_preserves_the_integer_cast_then_log1p_contract(int frames)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var scores = new Tensor<Half>(new[] { 1, frames });
            var path = TtsModelBase<Half>.MonotonicAlignmentPath(engine, scores);
            Assert.All(path.ToArray(), value => Assert.Equal((Half)1, value));
            var count = engine.ReduceSum(path, new[] { 1 }, false);
            Assert.Equal((Half)frames, count.ToArray()[0]);
            var target = engine.TensorLog1p(count);
            Assert.Equal((Half)Math.Log(1 + (double)(Half)frames), target.ToArray()[0]);
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
