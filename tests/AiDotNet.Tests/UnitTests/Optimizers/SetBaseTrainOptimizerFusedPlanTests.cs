using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Installing a different base optimizer after fused compiled training has committed must start a fresh
/// compiled plan. The plan's Adam moments belong to the old optimizer, and the committed-plan guard forbids an
/// eager fallback, so before the fix the next Train threw "Fused compiled training has already run successfully,
/// but the current step cannot engage the fused path".
/// </summary>
// Toggles the process-wide TensorCodecOptions.EnableCompilation, like the compiled-inference tests.
[Collection("NonParallelIntegration")]
public sealed class SetBaseTrainOptimizerFusedPlanTests
{
    [Fact]
    public void ReplacingTheOptimizerAfterAFusedStep_TrainsAgainOnAFreshFusedPlan()
    {
        bool saved = TensorCodecOptions.Current.EnableCompilation;
        try
        {
            TensorCodecOptions.Current.EnableCompilation = true;
            CompiledTapeTrainingStep<float>.Invalidate();
            var model = new Transformer<float>(TinyArchitecture(),
                lossFunction: new CategoricalCrossEntropyLoss<float>(),
                optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }));
            model.SetTrainingMode(true);
            var (input, target) = Data();

            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            model.Train(input, target);
            model.Train(input, target);
            long fusedBeforeSwap = CompiledTapeTrainingStep<float>.GetFusedStepCount();
            // Positive control: the fused plan really committed, so the swap below exercises the committed path.
            Assert.True(fusedBeforeSwap > 0, "the fused path never engaged, so this test would prove nothing");

            model.SetBaseTrainOptimizer(new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 5e-2 }));

            // Invalidating the plan also resets the global fused-step counter, so the post-swap phase is counted
            // from zero. A fresh plan traces on its first step and runs fused from the next, as before the swap.
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            var swapException = Record.Exception(() => { model.Train(input, target); model.Train(input, target); });
            Assert.Null(swapException);
            Assert.True(CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "the step after the swap fell off the fused path instead of building a fresh plan; last miss: " +
                (AiDotNet.Configuration.TrainingDiagnosticsConfig.LastFusedOptimizerMissReason ?? "(none recorded)"));
        }
        finally
        {
            TensorCodecOptions.Current.EnableCompilation = saved;
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    private static TransformerArchitecture<float> TinyArchitecture() => new(
        inputType: InputType.TwoDimensional,
        taskType: NeuralNetworkTaskType.SequenceClassification,
        numEncoderLayers: 1,
        numDecoderLayers: 0,
        numHeads: 2,
        modelDimension: 8,
        feedForwardDimension: 16,
        inputSize: 4,
        outputSize: 4,
        maxSequenceLength: 4,
        vocabularySize: 4,
        dropoutRate: 0.0,
        warmupSteps: 10,
        randomSeed: 42);

    private static (Tensor<float> Input, Tensor<float> Target) Data()
    {
        var input = new Tensor<float>([1, 4]);
        for (int s = 0; s < 4; s++) input[0, s] = s % 4;
        var target = new Tensor<float>([1, 4]);
        target[0, 1] = 1f;
        return (input, target);
    }
}
