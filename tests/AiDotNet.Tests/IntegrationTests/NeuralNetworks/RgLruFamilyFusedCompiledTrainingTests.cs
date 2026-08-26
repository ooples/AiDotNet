using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards the shared fused RG-LRU training route used by Griffin, Hawk, and RecurrentGemma. The
/// assertions require two real compiled updates so neither an opt-out nor a one-time eager fallback
/// can hide a broken replayed recurrence.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public sealed class RgLruFamilyFusedCompiledTrainingTests
{
    [Fact(Timeout = 120000)]
    public async Task RecurrentGemma_FusedCompiledStepKeepsUpdatesFinite()
    {
        await Task.Yield();
        AssertFusedUpdate(
            "RecurrentGemma",
            () => new TestableRecurrentGemma(CreateArchitecture()));
    }

    [Fact(Timeout = 120000)]
    public async Task Griffin_FusedCompiledStepKeepsUpdatesFinite()
    {
        await Task.Yield();
        AssertFusedUpdate(
            "Griffin",
            () => new TestableGriffin(CreateArchitecture()));
    }

    [Fact(Timeout = 120000)]
    public async Task Hawk_FusedCompiledStepKeepsUpdatesFinite()
    {
        await Task.Yield();
        AssertFusedUpdate(
            "Hawk",
            () => new TestableHawk(CreateArchitecture()));
    }

    /// <summary>
    /// Finite check that exists on every target framework.
    /// </summary>
    /// <remarks>
    /// <c>double.IsFinite</c> arrived in .NET Core 2.1 and is absent from .NET Framework 4.7.1,
    /// which this project still targets, so calling it compiles for net10.0 and breaks the net471
    /// leg of the same build. The same one-line helper is defined by the other suites that need
    /// this check for the same reason.
    /// </remarks>
    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private static NeuralNetworkArchitecture<double> CreateArchitecture()
        => new(
            InputType.OneDimensional,
            NeuralNetworkTaskType.TextGeneration,
            inputSize: 32,
            outputSize: 128);

    private static void AssertFusedUpdate(
        string modelName,
        Func<IFusedTrainingProbe> createModel)
    {
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<double>.Invalidate();
            CompiledTapeTrainingStep<double>.ResetFusedStepCount();

            using var model = createModel();
            var recurrentLayers = ((ILayeredModel<double>)model).Layers
                .OfType<RealGatedLinearRecurrenceLayer<double>>()
                .ToArray();
            Assert.NotEmpty(recurrentLayers);

            var input = new Tensor<double>([32]);
            for (int i = 0; i < input.Length; i++) input[i] = i % 128;

            var prediction = model.Predict(input);
            var target = new Tensor<double>(prediction.Shape.ToArray());
            int rows = target.Length / 128;
            for (int row = 0; row < rows; row++)
                target[(row * 128) + ((row + 1) % 128)] = 1.0;

            var parametersBefore = model.GetParameters().ToArray();
            model.Train(input, target);
            var parametersAfter = model.GetParameters().ToArray();
            var gradients = model.GetParameterGradients().ToArray();

            Assert.True(
                CompiledTapeTrainingStep<double>.GetFusedStepCount() > 0,
                $"{modelName} did not exercise the fused compiled step. Every model must train "
                + "through the fused path; an opt-out would hide the real defect instead of fixing it.");
            Assert.False(model.FusedTrainingDisabled,
                $"{modelName} sticky-disabled fused training during its first compiled update.");
            Assert.True(
                parametersBefore.Where((value, index) => value != parametersAfter[index]).Any(),
                $"{modelName}'s fused compiled step did not update a live parameter.");
            Assert.True(IsFinite(model.GetLastLoss()),
                $"{modelName}'s fused compiled loss was not finite.");
            Assert.All(parametersAfter, value => Assert.True(IsFinite(value),
                $"{modelName} produced a non-finite parameter."));
            Assert.All(gradients, value => Assert.True(IsFinite(value),
                $"{modelName} published a non-finite gradient."));
            Assert.Contains(gradients, value => value != 0.0);

            // A second step proves the compiled plan replays correctly against the live recurrent
            // state rather than succeeding once and then drifting.
            long afterFirst = CompiledTapeTrainingStep<double>.GetFusedStepCount();
            model.Train(input, target);
            var parametersAfterSecond = model.GetParameters().ToArray();
            var gradientsAfterSecond = model.GetParameterGradients().ToArray();
            Assert.True(
                CompiledTapeTrainingStep<double>.GetFusedStepCount() > afterFirst,
                $"{modelName} stopped using the fused compiled step after the first update.");
            Assert.False(model.FusedTrainingDisabled,
                $"{modelName} sticky-disabled fused training while replaying the compiled graph.");
            Assert.True(IsFinite(model.GetLastLoss()),
                $"{modelName}'s replayed fused loss was not finite.");
            Assert.Contains(Enumerable.Range(0, parametersAfter.Length),
                index => parametersAfter[index] != parametersAfterSecond[index]);
            Assert.All(parametersAfterSecond, value => Assert.True(IsFinite(value),
                $"{modelName} produced a non-finite parameter on compiled replay."));
            Assert.All(gradientsAfterSecond, value => Assert.True(IsFinite(value),
                $"{modelName} published a non-finite gradient on compiled replay."));
            Assert.Contains(gradientsAfterSecond, value => value != 0.0);
        }
        finally
        {
            CompiledTapeTrainingStep<double>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    private interface IFusedTrainingProbe : INeuralNetworkModel<double>, IDisposable
    {
        bool FusedTrainingDisabled { get; }
    }

    private sealed class TestableRecurrentGemma : RecurrentGemmaLanguageModel<double>, IFusedTrainingProbe
    {
        internal TestableRecurrentGemma(NeuralNetworkArchitecture<double> architecture)
            : base(architecture, vocabSize: 128, modelDimension: 32,
                numLayers: 1, maxSeqLength: 32)
        {
        }

        public bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }

    private sealed class TestableGriffin : GriffinLanguageModel<double>, IFusedTrainingProbe
    {
        internal TestableGriffin(NeuralNetworkArchitecture<double> architecture)
            : base(architecture, vocabSize: 128, modelDimension: 32,
                numLayers: 1, maxSeqLength: 32,
                options: new GriffinOptions { RecurrenceDimension = 40 })
        {
        }

        public bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }

    private sealed class TestableHawk : HawkLanguageModel<double>, IFusedTrainingProbe
    {
        internal TestableHawk(NeuralNetworkArchitecture<double> architecture)
            : base(architecture, vocabSize: 128, modelDimension: 32,
                numLayers: 1, maxSeqLength: 32,
                options: new HawkOptions { RecurrenceDimension = 40 })
        {
        }

        public bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }
}
