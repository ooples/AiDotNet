using System.Reflection;
using AiDotNet.Audio.Classification;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Finance.Graph;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

public class PaperTrainingReviewRegressionTests
{
    [Fact]
    public void AudioEventDetector_DefaultsToPaperMultiLabelLogitObjective()
    {
        var options = new AudioEventDetectorOptions
        {
            CustomLabels = ["Speech", "Music"]
        };

        using var model = new AudioEventDetector<double>(
            CreateArchitecture(inputSize: 64, outputSize: options.CustomLabels.Length),
            options);

        var loss = typeof(NeuralNetworkBase<double>).GetField(
            "LossFunction",
            BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model);

        Assert.IsType<BinaryCrossEntropyWithLogitsLoss<double>>(loss);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void VisionTS_RejectsInvalidLearningRateBeforeCreatingOptimizer(double learningRate)
    {
        var options = new VisionTSOptions<double>
        {
            ContextLength = 8,
            ForecastHorizon = 2,
            PatchLength = 2,
            HiddenDimension = 4,
            NumLayers = 1,
            NumHeads = 1,
            IntermediateSize = 8,
            LearningRate = learningRate
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new VisionTS<double>(CreateArchitecture(inputSize: 8, outputSize: 2), options));

        Assert.Equal(nameof(VisionTSOptions<double>.LearningRate), exception.ParamName);
    }

    [Fact]
    public void DCRNN_RegistersItsPaperOptimizerWithSharedTraining()
    {
        var options = new DCRNNOptions<double>
        {
            SequenceLength = 2,
            ForecastHorizon = 1,
            NumNodes = 2,
            NumFeatures = 1,
            HiddenDimension = 2,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            DiffusionSteps = 1,
            NumSamples = 1
        };

        using var model = new DCRNN<double>(
            CreateArchitecture(inputSize: 4, outputSize: 2),
            options,
            new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } });

        var paperOptimizer = typeof(DCRNN<double>).GetField(
            "_optimizer",
            BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model);
        var getOptimizer = typeof(NeuralNetworkBase<double>).GetMethod(
            "GetOrCreateBaseOptimizer",
            BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(paperOptimizer);
        Assert.NotNull(getOptimizer);
        Assert.Same(paperOptimizer, getOptimizer.Invoke(model, null));
    }

    [Fact]
    public void TotemForecastingOptions_DoNotExposeInactiveCommitmentWeight()
    {
        Assert.Null(typeof(TOTEMOptions<double>).GetProperty("CommitmentWeight"));
        Assert.Null(typeof(TOTEM<double>).GetField(
            "_commitmentWeight",
            BindingFlags.Instance | BindingFlags.NonPublic));
    }

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int inputSize, int outputSize)
        => new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputSize,
            outputSize: outputSize);
}
