using AiDotNet.Interfaces;
using AiDotNet.TimeSeries.AnomalyDetection;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class TimeSeriesIsolationForestTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new TimeSeriesIsolationForest<double>();
}
