using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class GARCHModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new GARCHModel<double>();
}
