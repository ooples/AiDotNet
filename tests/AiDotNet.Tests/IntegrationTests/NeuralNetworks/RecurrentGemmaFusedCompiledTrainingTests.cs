using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards RecurrentGemma's real fused compiled training route. This intentionally checks more than
/// parameter finiteness: the fused-step counter and a live parameter change prevent either eager
/// fallback or the compiled plan's non-finite-gradient safety skip from producing a false pass.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public sealed class RecurrentGemmaFusedCompiledTrainingTests
{
    [Fact(Timeout = 120000)]
    public async Task Train_UsesFusedCompiledStepAndAppliesAFiniteUpdate()
    {
        await Task.Yield();
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            using var model = new TestableRecurrentGemma(
                new NeuralNetworkArchitecture<float>(
                    InputType.OneDimensional,
                    NeuralNetworkTaskType.TextGeneration,
                    inputSize: 16,
                    outputSize: 16),
                vocabSize: 16,
                modelDimension: 8,
                numLayers: 1,
                maxSeqLength: 4);

            var input = new Tensor<float>([4]);
            var target = new Tensor<float>([4]);
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = i;
                target[i] = (i + 1) % 4;
            }

            var parametersBefore = model.GetParameters().ToArray();

            model.Train(input, target);

            Assert.True(
                CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "RecurrentGemma fell back to eager training instead of executing its fused compiled step.");
            Assert.False(
                model.FusedTrainingDisabled,
                "RecurrentGemma's fused step failed and sticky-disabled compilation before eager fallback.");

            var parametersAfter = model.GetParameters().ToArray();
            Assert.True(
                parametersBefore.Where((value, index) => value != parametersAfter[index]).Any(),
                "The fused plan did not update any live parameter; a rejected non-finite step must not count as success.");
            Assert.True(float.IsFinite(model.GetLastLoss()), "The fused training loss was not finite.");
            Assert.All(parametersAfter, value => Assert.True(float.IsFinite(value), "A fused parameter update was not finite."));
        }
        finally
        {
            CompiledTapeTrainingStep<float>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    private sealed class TestableRecurrentGemma : RecurrentGemmaLanguageModel<float>
    {
        internal TestableRecurrentGemma(
            NeuralNetworkArchitecture<float> architecture,
            int vocabSize,
            int modelDimension,
            int numLayers,
            int maxSeqLength)
            : base(architecture, vocabSize, modelDimension, numLayers, maxSeqLength)
        {
        }

        internal bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }
}
