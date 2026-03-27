using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class NBEATSModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new NBEATSModel<double>(new AiDotNet.Models.Options.NBEATSModelOptions<double>
        {
            NumStacks = 2,
            NumBlocksPerStack = 1,
            LookbackWindow = 10,
            ForecastHorizon = 5,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            MaxTrainingTimeSeconds = 5
        });
}
