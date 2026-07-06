using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class VectorAutoRegressionModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        // Lag = 1 is the well-specified model for the trended test series. With a deterministic
        // trend, consecutive lags y[t-1], y[t-2], ... are near-collinear, so an AR(2)+ OLS fit is
        // ill-conditioned: the near-unit-root weight is split across collinear lags and the fitted
        // characteristic polynomial lands inside the unit circle, producing a mean-reverting (and
        // therefore worse-than-naive) forecast. AR(1) captures the trend cleanly through its
        // unit-root drift (phi ~= 1, intercept ~= the per-step change), which is exactly why the
        // VARMA model (default Lag = 1) tracks this same series. Differencing the data would be the
        // alternative way to support higher lags on a trend.
        => new VectorAutoRegressionModel<double>(new AiDotNet.Models.Options.VARModelOptions<double>
        {
            OutputDimension = 1,
            Lag = 1
        });
}
