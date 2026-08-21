using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards the Griffin, Hawk, and RecurrentGemma fused compiled training routes. This intentionally
/// checks more than parameter finiteness: the fused-step counter and a live parameter change prevent
/// either eager fallback or the compiled plan's non-finite-gradient safety skip from producing a false pass.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public sealed class RgLruFamilyFusedCompiledTrainingTests
{
    [Fact(Timeout = 120000)]
    public async Task RecurrentGemma_UsesFusedCompiledStepAndAppliesAFiniteUpdate()
    {
        await Task.Yield();
        AssertFusedUpdate(
            "RecurrentGemma",
            () => new TestableRecurrentGemma(CreateArchitecture()));
    }

    [Fact(Timeout = 120000)]
    public async Task Griffin_UsesFusedCompiledStepAndAppliesAFiniteUpdate()
    {
        await Task.Yield();
        AssertFusedUpdate(
            "Griffin",
            () => new TestableGriffin(CreateArchitecture()));
    }

    [Fact(Timeout = 120000)]
    public async Task Hawk_UsesFusedCompiledStepAndAppliesAFiniteUpdate()
    {
        await Task.Yield();
        AssertFusedUpdate(
            "Hawk",
            () => new TestableHawk(CreateArchitecture()));
    }

    private static NeuralNetworkArchitecture<float> CreateArchitecture()
        => new(
            InputType.OneDimensional,
            NeuralNetworkTaskType.TextGeneration,
            inputSize: 16,
            outputSize: 16);

    private static void AssertFusedUpdate(string modelName, Func<IFusedTrainingProbe> createModel)
    {
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            using var model = createModel();

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
                $"{modelName} fell back to eager training instead of executing its fused compiled step.");
            Assert.False(
                model.FusedTrainingDisabled,
                $"{modelName}'s fused step failed and sticky-disabled compilation before eager fallback.");

            var parametersAfter = model.GetParameters().ToArray();
            Assert.True(
                parametersBefore.Where((value, index) => value != parametersAfter[index]).Any(),
                "The fused plan did not update any live parameter; a rejected non-finite step must not count as success.");
            Assert.True(IsFinite(model.GetLastLoss()), "The fused training loss was not finite.");
            Assert.All(parametersAfter, value => Assert.True(IsFinite(value), "A fused parameter update was not finite."));
        }
        finally
        {
            CompiledTapeTrainingStep<float>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    private static bool IsFinite(float value)
        => !float.IsNaN(value) && !float.IsInfinity(value);

    private interface IFusedTrainingProbe : AiDotNet.Interfaces.INeuralNetworkModel<float>, IDisposable
    {
        bool FusedTrainingDisabled { get; }
    }

    private sealed class TestableRecurrentGemma : RecurrentGemmaLanguageModel<float>, IFusedTrainingProbe
    {
        internal TestableRecurrentGemma(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, vocabSize: 16, modelDimension: 8, numLayers: 1, maxSeqLength: 4)
        {
        }

        public bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }

    private sealed class TestableGriffin : GriffinLanguageModel<float>, IFusedTrainingProbe
    {
        internal TestableGriffin(NeuralNetworkArchitecture<float> architecture)
            : base(
                architecture,
                vocabSize: 16,
                modelDimension: 8,
                numLayers: 1,
                maxSeqLength: 4,
                options: new GriffinOptions { RecurrenceDimension = 8 })
        {
        }

        public bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }

    private sealed class TestableHawk : HawkLanguageModel<float>, IFusedTrainingProbe
    {
        internal TestableHawk(NeuralNetworkArchitecture<float> architecture)
            : base(
                architecture,
                vocabSize: 16,
                modelDimension: 8,
                numLayers: 1,
                maxSeqLength: 4,
                options: new HawkOptions { RecurrenceDimension = 8 })
        {
        }

        public bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }
}
