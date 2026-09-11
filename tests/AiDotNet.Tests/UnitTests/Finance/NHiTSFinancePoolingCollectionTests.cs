using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Neural;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Regression tests for the N-HiTS training-path (tape) pooling. It documented trimming the
/// incomplete trailing window like the inference pooling does, but instead collected the trimmed
/// slices into a list that was never used and then threw, so any lookback window that is not a
/// multiple of every stack's pooling kernel could be predicted with but never trained.
/// </summary>
public class NHiTSFinancePoolingCollectionTests
{
    private static NHiTSFinance<double> CreateModel(int lookback, int horizon)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: lookback,
            outputSize: horizon);

        var options = new NHiTSOptions<double>
        {
            LookbackWindow = lookback,
            ForecastHorizon = horizon,
            HiddenLayerSize = 8,
            NumHiddenLayers = 2,
            DropoutRate = 0.0,
            // 12 is not a multiple of the first stack's kernel (8): the tape path must pool the
            // first 8 steps and drop the trailing 4, exactly as the inference pooling does.
            PoolingKernelSizes = new[] { 8, 4, 1 }
        };

        return new NHiTSFinance<double>(architecture, options);
    }

    private static Tensor<double> Ramp(int length)
    {
        var tensor = new Tensor<double>(new[] { 1, length });
        for (int i = 0; i < length; i++)
            tensor[0, i] = 0.1 * i;
        return tensor;
    }

    [Fact(Timeout = 60000)]
    public async Task Train_LookbackNotAMultipleOfPoolingKernel_DoesNotThrow()
    {
        var model = CreateModel(lookback: 12, horizon: 4);

        // Previously threw NotSupportedException("N-HiTS tape pooling: seqLen (12) must be a
        // multiple of kernelSize (8)...").
        var exception = Record.Exception(() => model.Train(Ramp(12), Ramp(4)));

        Assert.Null(exception);

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task ForwardForTraining_LookbackNotAMultipleOfPoolingKernel_ProducesFiniteForecast()
    {
        var model = CreateModel(lookback: 12, horizon: 4);

        var output = model.ForwardForTraining(Ramp(12));

        Assert.Equal(4, output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            double value = output.GetFlat(i);
            Assert.False(double.IsNaN(value) || double.IsInfinity(value), $"Non-finite forecast at {i}.");
        }

        await Task.CompletedTask;
    }
}
