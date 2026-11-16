using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class VitsAlignmentReplayTests
{
    [Theory]
    [InlineData(2049)]
    [InlineData(4097)]
    public void Half_alignment_keeps_binary_paths_and_rounds_long_duration_counts_only_once(int frames)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var mean = new Tensor<Half>(new[] { (Half)20.0 }, new[] { 1, 1, 1 });
            var logScale = new Tensor<Half>(new[] { (Half)0.0 }, new[] { 1, 1, 1 });
            var latent = new Tensor<Half>(Enumerable.Repeat((Half)20.0, frames).ToArray(), new[] { 1, 1, frames });
            var path = VitsTtsModelBase<Half>.AlignmentPath(engine, mean, logScale, latent, int.MaxValue, 0);
            Assert.Equal(new[] { 1, frames }, path.Shape.ToArray());
            Assert.All(path.ToArray(), value => Assert.Equal((Half)1.0, value));
            var counts = engine.ReduceSum(path, new[] { 1 }, false);
            Assert.Equal((Half)frames, counts.ToArray()[0]);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    // For two tokens the complete monotonic search is independently enumerable:
    // token zero owns a nonempty prefix; token one owns the remaining frames.
    private static double[] Oracle(double[] mean, double[] logScale, double[] latent, int seed, double scale)
    {
        var scores = new double[8];
        for (int token = 0; token < 2; token++)
            for (int frame = 0; frame < 4; frame++)
            {
                double delta = latent[frame] - mean[token];
                scores[token * 4 + frame] = -0.5 * Math.Log(2 * Math.PI) - logScale[token]
                    - 0.5 * delta * delta * Math.Exp(-2 * logScale[token]);
            }
        if (scale > 0)
        {
            double average = scores.Average();
            double std = Math.Sqrt(scores.Sum(value => (value - average) * (value - average)) / 7);
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
            for (int i = 0; i < scores.Length; i++)
            {
                double u1 = 1 - random.NextDouble(), u2 = random.NextDouble();
                scores[i] += std * scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }
        }
        int bestSplit = 1;
        double best = double.NegativeInfinity;
        for (int split = 1; split < 4; split++)
        {
            double value = 0;
            for (int frame = 0; frame < 4; frame++) value += scores[(frame < split ? 0 : 4) + frame];
            if (value > best) { best = value; bestSplit = split; }
        }
        var path = new double[8];
        for (int frame = 0; frame < 4; frame++) path[(frame < bestSplit ? 0 : 4) + frame] = 1;
        return path;
    }

    [Fact]
    public void Compiled_alignment_rebinds_current_latents_and_metadata_and_scatters_expansion_gradients()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var weights = new Tensor<double>(new[] { 2.0, 5.0 }, new[] { 1, 2 });
            var parameters = new[] { weights };
            var frameWeights = new Tensor<double>(new[] { 1.0, 2.0, 4.0, 8.0 }, new[] { 1, 1, 4 });
            Tensor<double> Objective(double[] means, double[] scales, double[] latents, int seed, double noise)
            {
                var mean = new Tensor<double>(means, new[] { 1, 1, 2 });
                var logScale = new Tensor<double>(scales, new[] { 1, 1, 2 });
                var zp = new Tensor<double>(latents, new[] { 1, 1, 4 });
                var path = VitsTtsModelBase<double>.AlignmentPath(engine, mean, logScale, zp, seed, noise);
                var expanded = VitsTtsModelBase<double>.ExpandAligned(engine,
                    engine.Reshape(weights, new[] { 1, 1, 2 }), path);
                return engine.ReduceSum(engine.TensorMultiply(expanded, frameWeights), null);
            }
            var cases = new[]
            {
                (Mean: new[] { -2.0, 2.0 }, Log: new[] { 0.0, 0.0 }, Z: new[] { -2.0, -1.5, -1.0, 2.0 }, Seed: 16777217, Noise: 0.0),
                (Mean: new[] { -2.0, 2.0 }, Log: new[] { 0.0, 0.0 }, Z: new[] { -2.0, 1.0, 1.5, 2.0 }, Seed: int.MaxValue, Noise: 0.0),
                (Mean: new[] { -1.0, 1.0 }, Log: new[] { 0.2, -0.3 }, Z: new[] { -1.0, -0.2, 0.2, 1.0 }, Seed: 16777217, Noise: 3.0),
                (Mean: new[] { -1.0, 1.0 }, Log: new[] { 0.2, -0.3 }, Z: new[] { -1.0, -0.2, 0.2, 1.0 }, Seed: int.MaxValue, Noise: 3.0000000000000004),
                (Mean: new[] { -1.0, 1.0 }, Log: new[] { 0.2, -0.3 }, Z: new[] { -1.0, -0.2, 0.2, 1.0 }, Seed: int.MaxValue, Noise: 1e-123)
            };
            using var recording = ClosureObjectiveGuard<double>.BeginRecording();
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var first = cases[0];
                var loss = Objective(first.Mean, first.Log, first.Z, first.Seed, first.Noise);
                plan = scope.CompileTraining(parameters, loss);
            }
            using (plan)
            {
                var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
                Assert.True(guard != null, reason);
                Assert.True(guard!.HasForeignLeaves);
                guard.AttachTo(plan);
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.125);
                var expectedWeights = new[] { 2.0, 5.0 };
                foreach (var item in cases)
                {
                    Assert.Equal(ClosureGuardResult.Matched,
                        guard.TryRebind(() => Objective(item.Mean, item.Log, item.Z, item.Seed, item.Noise), parameters, out _));
                    var path = Oracle(item.Mean, item.Log, item.Z, item.Seed, item.Noise);
                    var gradient = new double[2];
                    for (int token = 0; token < 2; token++)
                        for (int frame = 0; frame < 4; frame++) gradient[token] += path[token * 4 + frame] * frameWeights[0, 0, frame];
                    double expectedLoss = gradient[0] * expectedWeights[0] + gradient[1] * expectedWeights[1];
                    Assert.Equal(expectedLoss, plan.Step().ToArray()[0], 10);
                    Assert.Equal(gradient, plan.Gradients[0].ToArray());
                    for (int token = 0; token < 2; token++) expectedWeights[token] -= 0.125 * gradient[token];
                    Assert.Equal(expectedWeights, weights.ToArray());
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
