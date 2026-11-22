using System;
using AiDotNet.Document.Options;
using AiDotNet.Document.PixelToSequence;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Base-harness controlProbe and automatic divergence detection isolate Pix2Struct's strip patch-budget probe without a manual ControlGroup entry.</summary>
public class Pix2StructOptionControlTests
{
    private sealed class StripFirst { }
    private sealed class SquareFirst { }

    private static Pix2Struct<double> Create() => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 32)
        {
            RandomSeed = 1337,
        },
        options: new Pix2StructOptions
        {
            ImageSize = 32,
            PatchSize = 16,
            MaxPatches = 4,
            MaxSequenceLength = 8,
            HiddenDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumHeads = 2,
            VocabSize = 32,
            Seed = 1337,
        });

    private static Tensor<double> Input()
    {
        var rng = RandomHelper.CreateSeededRandom(23);
        var tensor = new Tensor<double>(new[] { 1, 3, 32, 32 });
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble();
        return tensor;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MaxPatchesStripProbe_DoesNotPoisonSquareImageProbes_InEitherOrder(bool stripFirst)
    {
        Type fixture = stripFirst ? typeof(StripFirst) : typeof(SquareFirst);
        using var shell = Create();
        var input = Input();

        Assert.Equal(string.Empty, OptionSensitivityFixtureSetup.ControlGroup(shell, "MaxPatches"));

        OptionSensitivityProbe<INeuralNetworkModel<double>> BuildProbe(string probeOption) => new(
            Create,
            model => OptionSensitivityFixtureSetup.Observe(model, input, probeOption, (network, image) =>
            {
                network.SetTrainingMode(false);
                return OptionSensitivityHarness.Fingerprint(network.Predict(image).ToArray());
            }),
            train: null,
            prepareOptions: options => OptionSensitivityFixtureSetup.PrepareOptions(options, probeOption),
            controlGroup: string.Empty);

        var controlProbe = BuildProbe(string.Empty);
        OptionSensitivityReport Measure(string option)
            => OptionSensitivityHarness.Measure(fixture, option, BuildProbe(option), controlProbe);

        if (!stripFirst)
            Assert.Equal(OptionSensitivityOutcome.Insensitive, Measure("MaxGradNorm").Outcome);
        Assert.DoesNotContain(OptionSensitivityHarness.MeasuredControlGroups(fixture), group => group.EndsWith(":MaxPatches", StringComparison.Ordinal));
        Assert.Equal(OptionSensitivityOutcome.Sensitive, Measure("MaxPatches").Outcome);
        // The verdict alone does not show which control it was measured against: compared with the square control the
        // strip candidate reads as sensitive too. The harness must have measured MaxPatches against a control of its
        // own, built by the MaxPatches probe (the unperturbed strip), and no manual ControlGroup entry asked for it.
        Assert.Contains(OptionSensitivityHarness.MeasuredControlGroups(fixture),
            group => group.StartsWith("auto-", StringComparison.Ordinal) && group.EndsWith(":MaxPatches", StringComparison.Ordinal));
        Assert.Equal(OptionSensitivityOutcome.Insensitive, Measure("MaxGradNorm").Outcome);
        // MaxGradNorm observes the square image: it shares the default control and takes none of its own.
        Assert.Contains(string.Empty, OptionSensitivityHarness.MeasuredControlGroups(fixture));
        Assert.DoesNotContain(OptionSensitivityHarness.MeasuredControlGroups(fixture), group => group.EndsWith(":MaxGradNorm", StringComparison.Ordinal));
    }
}
