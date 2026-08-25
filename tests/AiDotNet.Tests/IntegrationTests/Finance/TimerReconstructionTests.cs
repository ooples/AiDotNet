using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>Regression coverage for Timer configuration reconstruction.</summary>
public sealed class TimerReconstructionTests
{
    [Fact(Timeout = 30000)]
    public async Task DeepCopy_PreservesConfiguredPaperOptimizerSettings()
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        architecture.RandomSeed = 4321;
        var options = new TimerOptions<double>
        {
            ContextLength = 4,
            ForecastHorizon = 2,
            PatchLength = 2,
            PatchStride = 2,
            HiddenDimension = 4,
            NumLayers = 1,
            NumHeads = 1,
            DropoutRate = 0,
            MaxContextLength = 4,
            LearningRate = 1.25e-4,
            WeightDecay = 0.031,
            LearningRateDecay = 0.73,
            Seed = 4321,
        };

        using var model = new Timer<double>(architecture, options);
        using var clone = Assert.IsType<Timer<double>>(model.DeepCopy());

        var cloneOptions = Assert.IsType<TimerOptions<double>>(clone.GetOptions());
        Assert.Equal(options.LearningRate, cloneOptions.LearningRate);
        Assert.Equal(options.WeightDecay, cloneOptions.WeightDecay);
        Assert.Equal(options.LearningRateDecay, cloneOptions.LearningRateDecay);
        Assert.Equal(options.Seed, cloneOptions.Seed);
    }
}
