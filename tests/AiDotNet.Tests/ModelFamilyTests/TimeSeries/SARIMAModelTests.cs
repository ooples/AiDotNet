using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class SARIMAModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new SARIMAModel<double>(new AiDotNet.Models.Options.SARIMAOptions<double>
        {
            SeasonalPeriod = 20 // Match test data's seasonal period
        });
}
