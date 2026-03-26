using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class STLDecompositionTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new STLDecomposition<double>(new AiDotNet.Models.Options.STLDecompositionOptions<double>
        {
            SeasonalPeriod = 20 // Match test data's seasonal period
        });
}
