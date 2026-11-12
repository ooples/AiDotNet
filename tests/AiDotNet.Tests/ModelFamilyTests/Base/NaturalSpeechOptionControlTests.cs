using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Real native frontend and waveform probes must not share a cached control observation.</summary>
public class NaturalSpeechOptionControlTests
{
    private sealed class FrontendFirst { }
    private sealed class WaveformFirst { }

    private static NaturalSpeech<double> Create() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 11 }, NaturalSpeechNativeOptionsTests.Tiny());

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void FrontendObservation_DoesNotMakeUnrelatedWaveformOptionsSensitive_InEitherOrder(bool frontendFirst)
    {
        Type fixture = frontendFirst ? typeof(FrontendFirst) : typeof(WaveformFirst);
        using var shell = Create();
        var input = shell.TextToTokens("!");
        OptionSensitivityReport Measure(string option) => OptionSensitivityHarness.Measure(fixture, option,
            new OptionSensitivityProbe<NaturalSpeech<double>>(Create,
                model => OptionSensitivityFixtureSetup.Observe(model, input, option, (network, tokens) =>
                {
                    network.SetTrainingMode(false);
                    return OptionSensitivityHarness.Fingerprint(network.Predict(tokens).ToArray());
                }), train: null,
                controlGroup: OptionSensitivityFixtureSetup.ControlGroup(shell, option)));

        if (!frontendFirst)
            Assert.Equal(OptionSensitivityOutcome.Insensitive, Measure("MaxMelLength").Outcome);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, Measure("AddBlank").Outcome);
        Assert.Equal(OptionSensitivityOutcome.Insensitive, Measure("MaxMelLength").Outcome);
    }
}
