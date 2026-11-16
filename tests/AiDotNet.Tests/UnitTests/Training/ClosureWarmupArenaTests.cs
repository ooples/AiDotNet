using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("EngineCurrentGlobalState")]
public sealed class ClosureWarmupArenaTests
{
    [Fact]
    public void Warmup_PreservesCapturedArenaBatch_ForTheCompiledUpdate()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            AssertWarmupArenaContract();
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private static void AssertWarmupArenaContract()
    {
        var engine = AiDotNetEngine.Current;
        var parameter = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { 0.25f }));
        var sourceValues = Enumerable.Range(0, 128).Select(i => 0.01f * (i / 4) + 0.1f * (i % 4)).ToArray();
        var source = new Tensor<float>(new[] { 32, 4 }, new Vector<float>(sourceValues));
        var expected = Enumerable.Range(0, 32).SelectMany(row =>
            Enumerable.Range(1, 3).Select(column => sourceValues[row * 4 + column])).ToArray();
        var optimizer = new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.05,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });
        var loss = new MeanAbsoluteErrorLoss<float>();
        using var arena = TensorArena.Create();
        var target = engine.TensorSlice(source, new[] { 0, 1 }, new[] { 32, 3 });

        float value = TapeTrainingStepper<float>.EagerObjectiveStep(new[] { parameter },
            () => loss.ComputeTapeLoss(engine.TensorMultiply(target,
                engine.TensorBroadcastTo(parameter, new[] { 32, 3 })), target), optimizer);

        Assert.True(CompiledTapeTrainingStep<float>.LastClosureCompileCount > 0);
        Assert.Equal(0, CompiledTapeTrainingStep<float>.LastClosureOutsidePlanSteps);
        // Reusing a same-sized flat scratch buffer must not reissue the still-live captured batch.
        var scratch = engine.TensorMultiplyScalar(new Tensor<float>(new[] { 96 }), 2f);
        Assert.Equal(1, scratch.Rank);
        Assert.Equal(2, target.Rank);
        Assert.Equal(32, target.Shape[0]);
        Assert.Equal(3, target.Shape[1]);
        Assert.Equal(expected, target.ToArray());
        double mean = expected.Select(x => (double)x).Average();
        Assert.Equal(0.75 * mean, value, 6);
        Assert.Equal(0.25 + 0.05 * mean, parameter[0], 6);
    }
}
