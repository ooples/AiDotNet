using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class TimeSeriesRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new TimeSeriesRegression<double>();

    /// <summary>
    /// This model fits an inner time series estimator (ARIMA by default) over a design matrix built
    /// from lagged targets and lagged predictors. The harness generates independent cross-sectional
    /// rows with no temporal ordering, so the lag structure the model exists to exploit carries no
    /// signal, and held-out R-squared and per-feature coefficient recovery measure that mismatch
    /// rather than the estimator. Its forecasting behaviour is exercised by the time series suites.
    /// </summary>
    protected override bool PredictiveQualityInvariantsApplicable => false;
}
