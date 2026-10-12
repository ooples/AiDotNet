using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class DeepARModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new DeepARModel<double>(new AiDotNet.Models.Options.DeepAROptions<double>
        {
            LookbackWindow = 20,
            ForecastHorizon = 5,
            HiddenSize = 16,
            NumLayers = 1,
            Epochs = 40,
            BatchSize = 8
        });
}
