using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class ARIMAModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ARIMAModel<double>();
}
