using AiDotNet.Interfaces;
using AiDotNet.TimeSeries.AnomalyDetection;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class LSTMVAETests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LSTMVAE<double>();

    // LSTMVAE is an anomaly detector — returns reconstruction errors, not forecasts
    protected override bool IsForecastingModel => false;
    protected override bool CanCaptureTrend => false;
}
