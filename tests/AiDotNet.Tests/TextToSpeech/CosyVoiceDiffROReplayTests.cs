using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class CosyVoiceDiffROReplayTests
{
    [Fact]
    public void Compiled_kl_uses_current_reference_logits_without_updating_the_reference()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var policy = new Tensor<double>(new[] { 0.2, -0.4, 0.7, -0.1, 0.5, 0.3 }, new[] { 6 });
            var reference = new Tensor<double>(new[] { 0.1, 0.3, -0.2, 0.4, -0.5, 0.2 }, new[] { 6 });
            var parameters = new[] { policy, reference };
            var input = new Tensor<double>(new[] { 1.0, 0.5, -1.0, 2.0, 0.75, -0.5 }, new[] { 2, 3 });
            const double beta = 0.35;
            const double learningRate = 0.125;
            Tensor<double> Objective()
            {
                var logits = engine.TensorMultiply(engine.Reshape(policy, new[] { 2, 3 }), input);
                var referenceLogits = engine.TensorMultiply(engine.Reshape(reference, new[] { 2, 3 }), input);
                return CosyVoiceDiffRO<double>.Objective(engine, logits, referenceLogits,
                    soft => engine.ReduceSum(engine.TensorMultiplyScalar(soft, 0.0), new[] { 0, 1 }, false),
                    new CosyVoiceDiffROOptions { KlCoefficient = beta }, new Random(17)).Loss;
            }

            using var recording = ClosureObjectiveGuard<double>.BeginRecording();
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
                plan = scope.CompileTraining(parameters, Objective());
            using (plan)
            {
                var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
                Assert.True(guard is not null, reason);
                Assert.True(guard!.LeafCount > 0);
                guard.AttachTo(plan);
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: learningRate);
                var expectedPolicy = policy.ToArray();
                for (int step = 0; step < 3; step++)
                {
                    // Both reference weights and foreign input change after compilation.
                    var currentReference = new double[6];
                    var currentInput = new double[6];
                    for (int i = 0; i < 6; i++)
                    {
                        currentReference[i] = (i - 2.0) * (step + 1) / 7;
                        reference[i] = currentReference[i];
                        currentInput[i] = (i % 2 == 0 ? 1 : -1) * (step + i + 1.0) / 4;
                    }
                    input = new Tensor<double>(currentInput, new[] { 2, 3 });
                    var gradient = new double[6];
                    double expectedLoss = 0;
                    for (int row = 0; row < 2; row++)
                    {
                        var p = new double[3];
                        var q = new double[3];
                        double sumP = 0, sumQ = 0;
                        for (int j = 0; j < 3; j++)
                        {
                            int i = row * 3 + j;
                            sumP += p[j] = Math.Exp(expectedPolicy[i] * currentInput[i]);
                            sumQ += q[j] = Math.Exp(currentReference[i] * currentInput[i]);
                        }
                        double rowKl = 0;
                        for (int j = 0; j < 3; j++)
                        {
                            p[j] /= sumP;
                            q[j] /= sumQ;
                            rowKl += p[j] * Math.Log(p[j] / q[j]);
                        }
                        expectedLoss += beta * rowKl;
                        for (int j = 0; j < 3; j++)
                        {
                            int i = row * 3 + j;
                            gradient[i] = beta * p[j] * (Math.Log(p[j] / q[j]) - rowKl) * currentInput[i];
                        }
                    }
                    Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                    Assert.Equal(expectedLoss, plan.Step().ToArray()[0], 10);
                    Assert.All(plan.Gradients[1].ToArray(), value => Assert.Equal(0.0, value));
                    Assert.Equal(currentReference, reference.ToArray());
                    var actualGradient = plan.Gradients[0].ToArray();
                    var actualPolicy = policy.ToArray();
                    for (int i = 0; i < 6; i++)
                    {
                        Assert.Equal(gradient[i], actualGradient[i], 10);
                        expectedPolicy[i] -= learningRate * gradient[i];
                        Assert.Equal(expectedPolicy[i], actualPolicy[i], 10);
                    }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
