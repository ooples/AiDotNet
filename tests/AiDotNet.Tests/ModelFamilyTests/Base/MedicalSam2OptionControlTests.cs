using System;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Base-harness controlProbe and automatic divergence detection isolate multi-image memory-bank probes without a manual ControlGroup entry.</summary>
public class MedicalSam2OptionControlTests
{
    private sealed class BankFirst { }
    private sealed class SingleFrameFirst { }

    private static MedicalSAM2<double> Create() => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 1)
        {
            RandomSeed = 1337,
        },
        options: new MedicalSAM2Options
        {
            ImageSize = 64,
            MemoryBankCapacity = 4,
            HieraEmbeddingDimension = 16,
            HieraStageDepths = new[] { 1, 1, 1, 1 },
            HieraInitialHeadCount = 1,
            HieraWindowSizes = new[] { 8, 4, 2, 1 },
            HieraGlobalAttentionBlockIndexes = new[] { 2 },
            ModelDimension = 16,
            MemoryDimension = 8,
            DecoderHeadCount = 4,
            MemoryAttentionLayerCount = 4,
            MemoryAttentionFeedForwardDimension = 64,
            MemoryFuserDimension = 32,
            MaskDecoderMlpDimension = 64,
            WarmupSteps = 0,
            LearningRate = 1e-3,
            ImageEncoderLearningRate = 1e-3,
            CorrectionPoints = 1,
            MemoryBankSize = 2,
            Seed = 1337,
        });

    private static Tensor<double> Input()
    {
        var rng = RandomHelper.CreateSeededRandom(19);
        var tensor = new Tensor<double>(new[] { 2, 3, 64, 64 });
        int frameLength = 3 * 64 * 64;
        for (int i = 0; i < tensor.Length; i++)
        {
            double pattern = 0.5 + 0.3 * Math.Sin(0.017 * (i % frameLength));
            tensor[i] = pattern + 0.1 * (rng.NextDouble() - 0.5);
        }
        return tensor;
    }

    private static Tensor<double> Target()
    {
        var mask = new Tensor<double>(new[] { 2, 64, 64 });
        for (int b = 0; b < 2; b++)
            for (int y = 16; y < 48; y++)
                for (int x = 16; x < 48; x++)
                    mask[b, y, x] = 1.0;
        return mask;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BaseHarnessControlProbe_AutomaticallyIsolatesCustomObservations_InEitherOrder(bool bankFirst)
    {
        Type fixture = bankFirst ? typeof(BankFirst) : typeof(SingleFrameFirst);
        using var shell = Create();
        var input = Input();
        var target = Target();

        Assert.Equal(string.Empty, OptionSensitivityFixtureSetup.ControlGroup(shell, "ConfidenceThreshold"));
        Assert.Equal(string.Empty, OptionSensitivityFixtureSetup.ControlGroup(shell, "MemoryResampling"));

        OptionSensitivityProbe<INeuralNetworkModel<double>> BuildProbe(string probeOption) => new(
            Create,
            model => OptionSensitivityFixtureSetup.Observe(model, input, probeOption, (network, frame) =>
            {
                network.SetTrainingMode(false);
                return OptionSensitivityHarness.Fingerprint(network.Predict(frame).ToArray());
            }),
            train: model => model.Train(input, target),
            prepareOptions: options => OptionSensitivityFixtureSetup.PrepareOptions(options, probeOption),
            controlGroup: string.Empty);

        var controlProbe = BuildProbe(string.Empty);
        OptionSensitivityReport Measure(string option)
            => OptionSensitivityHarness.Measure(fixture, option, BuildProbe(option), controlProbe);

        if (!bankFirst)
            Assert.Equal(OptionSensitivityOutcome.Sensitive, Measure("ModelDimension").Outcome);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, Measure("ConfidenceThreshold").Outcome);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, Measure("MemoryResampling").Outcome);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, Measure("ModelDimension").Outcome);
    }
}
