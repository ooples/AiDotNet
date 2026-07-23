using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class SpectralAnalysisModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new SpectralAnalysisModel<double>();

    // SpectralAnalysis is a frequency-domain tool, not a time-domain forecaster
    protected override bool IsForecastingModel => false;
    protected override bool CanCaptureTrend => false;
}
