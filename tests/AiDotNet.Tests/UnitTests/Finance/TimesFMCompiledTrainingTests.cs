using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// TimesFM's training step must stay on the compiled plan. RevIN's overflow check (#2588) read its statistics on the
/// host (TensorCountNonzero), which a graph trace refuses, so every step was abandoned to the eager tape: the census
/// warm step went 27 -> 55 ms and the fixture's allocations 148 -> 221 MB.
/// </summary>
[Collection("NonParallelIntegration")]
public class TimesFMCompiledTrainingTests
{
    [Fact(Timeout = 120000)]
    public async Task TrainStep_RunsOnTheCompiledPlan()
    {
        await Task.Yield();
        const int context = 64;
        const int horizon = 16;
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: context, outputSize: horizon)
        { RandomSeed = 7 };
        var options = new TimesFMOptions<float>
        {
            ContextLength = context,
            ForecastHorizon = horizon,
            PatchLength = 16,
            OutputPatchLength = 16,
            HiddenDimension = 32,
            NumLayers = 1,
            NumHeads = 2,
            DropoutRate = 0.0,
            UsePretrainedWeights = false,
            QuantileHeadDimension = 32,
        };
        var model = new TimesFM<float>(architecture, options);
        var input = new Tensor<float>(new[] { 1, context, 1 });
        for (int i = 0; i < context; i++) input[0, i, 0] = (float)System.Math.Sin(0.3 * i);
        var target = new Tensor<float>(new[] { 1, horizon });
        for (int i = 0; i < horizon; i++) target[0, i] = (float)System.Math.Sin(0.3 * (context + i));

        CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        model.Train(input, target);
        model.Train(input, target);

        Assert.Null(CompiledTapeTrainingStep<float>.GetLastFallbackException());
        Assert.True(CompiledTapeTrainingStep<float>.GetFusedStepCount() >= 2,
            $"fused steps {CompiledTapeTrainingStep<float>.GetFusedStepCount()}; last fallback: "
            + CompiledTapeTrainingStep<float>.GetLastFallbackException()?.Message);
    }
}
