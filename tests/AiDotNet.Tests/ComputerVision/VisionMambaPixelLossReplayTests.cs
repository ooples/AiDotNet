using System;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Mamba;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

[Collection("EngineCurrentGlobalState")]
public sealed class VisionMambaPixelLossReplayTests
{
    [Fact]
    public void Compiled_pixel_loss_replays_current_logits_and_targets_with_all_pixel_mean_and_weighted_gradients()
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            VerifyCompiledPixelLoss();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static void VerifyCompiledPixelLoss()
    {
        const int batches = 2, classes = 3, height = 2, width = 2, pixels = batches * height * width;
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<double>(new[] { batches, classes, height, width });
        var target = new Tensor<double>(new[] { batches, classes, height, width });
        var parameters = new[] { logits };
        double[] currentLogits = Array.Empty<double>(), currentTargets = Array.Empty<double>();
        void SetData(int step)
        {
            currentLogits = Enumerable.Range(0, logits.Length).Select(i =>
                1000 + step * 200 + 7 * Math.Sin(i * .37 + step * .61)).ToArray();
            currentTargets = new double[target.Length];
            for (int b = 0; b < batches; b++)
                for (int p = 0; p < height * width; p++)
                    for (int c = 0; c < classes; c++)
                        // Zero rows count in the denominator; other rows have varying, non-unit mass.
                        currentTargets[(b * classes + c) * height * width + p] = p == step % 4 ? 0
                            : (c == (p + b + step) % classes ? 1.7 : .15 * (step + 1));
            for (int i = 0; i < logits.Length; i++) logits.SetFlat(i, currentLogits[i]);
            // A fresh external leaf exercises closure rebinding, not just in-place data updates.
            target = new Tensor<double>(currentTargets, new[] { batches, classes, height, width });
        }
        Tensor<double> Objective() => VisionMamba<double>.PixelCrossEntropy(engine,
            engine.TensorPermute(logits, new[] { 0, 1, 3, 2 }), target);

        SetData(0);
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .125);
            for (int step = 0; step < 3; step++)
            {
                SetData(step + 1);
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                double expectedLoss = 0;
                var expectedGradient = new double[logits.Length];
                for (int b = 0; b < batches; b++)
                    for (int row = 0; row < height; row++)
                        for (int column = 0; column < width; column++)
                        {
                            int pixel = row * width + column;
                            int sourcePixel = column * height + row;
                            var scores = Enumerable.Range(0, classes).Select(c =>
                                currentLogits[(b * classes + c) * height * width + sourcePixel]).ToArray();
                            double maximum = scores.Max();
                            double sum = scores.Sum(value => Math.Exp(value - maximum));
                            double mass = Enumerable.Range(0, classes).Sum(c =>
                                currentTargets[(b * classes + c) * height * width + pixel]);
                            for (int c = 0; c < classes; c++)
                            {
                                double label = currentTargets[(b * classes + c) * height * width + pixel];
                                expectedLoss -= label * (scores[c] - maximum - Math.Log(sum)) / pixels;
                                expectedGradient[(b * classes + c) * height * width + sourcePixel] =
                                    (mass * Math.Exp(scores[c] - maximum) / sum - label) / pixels;
                            }
                        }
                double actualLoss = plan.Step().ToArray()[0];
                Assert.False(double.IsNaN(actualLoss) || double.IsInfinity(actualLoss));
                Assert.Equal(expectedLoss, actualLoss, 10);
                var gradient = plan.Gradients[0].ToArray();
                for (int i = 0; i < gradient.Length; i++)
                {
                    Assert.Equal(expectedGradient[i], gradient[i], 10);
                    Assert.Equal(currentLogits[i] - .125 * expectedGradient[i], logits.ToArray()[i], 10);
                }
            }
        }
    }
}
