using AiDotNet.Audio.Effects;
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Initialization;

public class OctaveBandpassInitializationStrategyTests
{
    [Fact]
    public void RoomImpulseResponseOptions_RejectsInvalidPaperLearningRate()
    {
        foreach (double learningRate in new[] { 0.0, -1.0, double.NaN, double.PositiveInfinity })
        {
            var options = new RoomImpulseResponseOptions { LearningRate = learningRate };
            Assert.Throws<ArgumentOutOfRangeException>(() => options.Validate());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RoomImpulseResponse_ValidatesOptionsBeforeConstructingLoss(bool onnxMode)
    {
        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 64,
            outputSize: 32);
        var options = new RoomImpulseResponseOptions
        {
            NumEncoderBlocks = 2,
            EncoderMaxChannels = 16,
            LatentDim = 8,
            NumDecoderBlocks = 2,
            NumNoiseBands = 3,
            NoiseFilterOrder = 15,
            RIRLength = 32,
            EarlyResponseLength = 8,
            StftFrameSizes = [0]
        };

        var exception = onnxMode
            ? Assert.Throws<ArgumentOutOfRangeException>(
                () => new RoomImpulseResponse<double>(architecture, "unused.onnx", options))
            : Assert.Throws<ArgumentOutOfRangeException>(
                () => new RoomImpulseResponse<double>(architecture, options));

        Assert.Equal(nameof(RoomImpulseResponseOptions.StftFrameSizes), exception.ParamName);
    }

    [Fact]
    public void InitializeWeights_CreatesSymmetricSelectiveUnitGainOctaveBands()
    {
        const int bandCount = 3;
        const int kernelSize = 64;
        var weights = new Tensor<double>([bandCount, 1, 1, kernelSize]);
        var strategy = new OctaveBandpassInitializationStrategy<double>();

        strategy.InitializeWeights(weights, kernelSize, bandCount);

        var centers = Enumerable.Range(0, bandCount)
            .Select(band => 0.5 / (Math.Sqrt(2.0) * Math.Pow(2.0, bandCount - 1 - band)))
            .ToArray();
        for (int band = 0; band < bandCount; band++)
        {
            int offset = band * kernelSize;
            double dcResponse = 0.0;
            for (int tap = 0; tap < kernelSize; tap++)
            {
                dcResponse += weights[offset + tap];
                Assert.Equal(weights[offset + tap], weights[offset + kernelSize - 1 - tap], 12);
            }

            Assert.InRange(Math.Abs(dcResponse), 0.0, 1e-12);
            Assert.InRange(FrequencyResponse(weights, offset, kernelSize, centers[band]), 0.999999999, 1.000000001);

            for (int otherBand = 0; otherBand < bandCount; otherBand++)
            {
                if (otherBand == band)
                {
                    continue;
                }

                Assert.InRange(
                    FrequencyResponse(weights, offset, kernelSize, centers[otherBand]),
                    0.0,
                    0.01);
            }
        }

        Assert.Equal(2.0, centers[1] / centers[0], 12);
        Assert.Equal(2.0, centers[2] / centers[1], 12);
    }

    [Fact]
    public void CreateDefaultRoomImpulseResponseLayers_UsesOctaveBandpassFilterbank()
    {
        const int bandCount = 3;
        const int filterOrder = 63;
        var layers = LayerHelper<double>.CreateDefaultRoomImpulseResponseLayers(
            numEncoderBlocks: 1,
            encoderMaxChannels: 16,
            latentDim: 8,
            numDecoderBlocks: 1,
            numNoiseBands: bandCount,
            noiseFilterOrder: filterOrder).ToList();

        var filterbank = Assert.IsType<Conv1DLayer<double>>(layers[^2]);
        var parameters = filterbank.GetParameters();
        int kernelSize = filterOrder + 1;

        for (int band = 0; band < bandCount; band++)
        {
            int offset = band * kernelSize;
            double dcResponse = 0.0;
            for (int tap = 0; tap < kernelSize; tap++)
            {
                dcResponse += parameters[offset + tap];
            }

            Assert.InRange(Math.Abs(dcResponse), 0.0, 1e-12);
        }
    }

    private static double FrequencyResponse(
        Tensor<double> weights,
        int offset,
        int kernelSize,
        double frequency)
    {
        double real = 0.0;
        double imaginary = 0.0;
        for (int tap = 0; tap < kernelSize; tap++)
        {
            double phase = -2.0 * Math.PI * frequency * tap;
            real += weights[offset + tap] * Math.Cos(phase);
            imaginary += weights[offset + tap] * Math.Sin(phase);
        }

        return Math.Sqrt(real * real + imaginary * imaginary);
    }
}
