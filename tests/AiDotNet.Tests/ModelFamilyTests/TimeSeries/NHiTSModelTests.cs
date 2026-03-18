using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class NHiTSModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new NHiTSModel<double>(new AiDotNet.Models.Options.NHiTSOptions<double>
        {
            NumStacks = 2,
            NumBlocksPerStack = 1,
            LookbackWindow = 20,
            ForecastHorizon = 5,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            PoolingKernelSizes = new[] { 4, 1 },
            PoolingModes = new[] { "MaxPool", "AvgPool" },
            InterpolationModes = new[] { "Linear", "Linear" }
        });
}
