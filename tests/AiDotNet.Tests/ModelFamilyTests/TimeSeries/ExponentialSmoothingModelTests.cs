using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class ExponentialSmoothingModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ExponentialSmoothingModel<double>(new AiDotNet.Models.Options.ExponentialSmoothingOptions<double>
        {
            UseSeasonal = true,
            SeasonalPeriod = 20, // Match test data's seasonal period
            IncludeTrend = true
        });
}
