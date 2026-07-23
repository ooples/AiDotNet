using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.StateSpace;
using AiDotNet.Initialization;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

public class HippoConfigurationTests
{
    [Fact]
    public void Defaults_MatchOfficialLegSRecurrentConfiguration()
    {
        var options = new HippoOptions<double>();

        Assert.Equal(1024, options.ContextLength);
        Assert.Equal(256, options.ModelDimension);
        Assert.Equal(-1, options.StateDimension);
        Assert.Equal(1, options.MemorySize);
        Assert.Equal(1, options.NumLayers);
        Assert.Equal(0.0, options.DropoutRate);
        Assert.Equal("legs", options.HippoMethod);
        Assert.Equal("bilinear", options.DiscretizationMethod);
        Assert.Equal(0, options.InitialTime);
        Assert.Equal(0.0, options.TimeStep);
        Assert.Equal(0.0, options.TimescaleMin);
        Assert.Equal(double.PositiveInfinity, options.TimescaleMax);
        Assert.True(options.UseGate);
        Assert.False(options.UseNormalization);

        using var model = new Hippo<double>(CreateArchitecture(1024, options.ForecastHorizon), options);
        var cell = Assert.IsType<HippoMemoryCellLayer<double>>(model.Layers[0]);
        Assert.Equal(256, model.StateDimension);
        Assert.Equal(256, cell.HiddenSize);
        Assert.Equal(256, cell.MemoryOrder);
        Assert.Equal(1, cell.MemorySize);
        Assert.Equal("legs", cell.Measure);
        Assert.Equal("bilinear", cell.Discretization);
    }

    [Fact]
    public void CustomOptions_PropagateToEveryRecurrentCell()
    {
        var options = new HippoOptions<double>
        {
            ContextLength = 12,
            ForecastHorizon = 3,
            ModelDimension = 16,
            StateDimension = 8,
            MemorySize = 2,
            NumLayers = 2,
            DropoutRate = 0.25,
            HippoMethod = "legt",
            DiscretizationMethod = "zoh",
            InitialTime = 3,
            TimeStep = 0.2,
            TimescaleMin = 0.01,
            TimescaleMax = 2.0,
            UseGate = false,
            UseNormalization = true
        };

        using var model = new Hippo<double>(CreateArchitecture(12, 3), options, numFeatures: 4);
        var cells = model.Layers.OfType<HippoMemoryCellLayer<double>>().ToArray();

        Assert.Equal(2, cells.Length);
        Assert.All(cells, cell =>
        {
            Assert.Equal(16, cell.HiddenSize);
            Assert.Equal(8, cell.MemoryOrder);
            Assert.Equal(2, cell.MemorySize);
            Assert.Equal("legt", cell.Measure);
            Assert.Equal("zoh", cell.Discretization);
            var metadata = cell.GetMetadata();
            Assert.Equal("3", metadata["InitialTime"]);
            Assert.Equal("0.2", metadata["TimeStep"]);
            Assert.Equal("0.01", metadata["TimescaleMin"]);
            Assert.Equal("2", metadata["TimescaleMax"]);
            Assert.Equal("False", metadata["UseGate"]);
        });
        Assert.Equal(2, model.Layers.OfType<LayerNormalizationLayer<double>>().Count());
        Assert.Equal(2, model.Layers.OfType<DropoutLayer<double>>().Count());

        var info = model.GetModelMetadata().AdditionalInfo;
        Assert.Equal(12, info["ContextLength"]);
        Assert.Equal(3, info["ForecastHorizon"]);
        Assert.Equal(16, info["ModelDimension"]);
        Assert.Equal(8, info["StateDimension"]);
        Assert.Equal(2, info["MemorySize"]);
        Assert.Equal(2, info["NumLayers"]);
        Assert.Equal("legt", info["HippoMethod"]);
        Assert.Equal("zoh", info["DiscretizationMethod"]);
        Assert.Equal(false, info["UseGate"]);
        Assert.Equal(true, info["UseNormalization"]);
    }

    [Fact]
    public void CustomInitializationStrategy_AppliesToAllLearnedProjections()
    {
        var cell = new HippoMemoryCellLayer<double>(
            hiddenSize: 4,
            inputSize: 2,
            memoryOrder: 3,
            initializationStrategy: new ZeroInitializationStrategy<double>());

        Assert.Equal(6, cell.GetTrainableParameters().Count);
        Assert.All(cell.GetParameters(), value => Assert.Equal(0.0, value));
    }

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int inputSize, int outputSize) =>
        new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputSize,
            outputSize: outputSize);
}
