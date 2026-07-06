namespace AiDotNet.Interfaces;

/// <summary>
/// Marks a time-series model that forecasts from <b>future exogenous regressors</b>
/// (e.g. ARIMAX, dynamic regression with ARIMA errors). The forecast horizon is the row
/// count of the supplied matrix, and each row carries the exogenous-variable values for
/// that future step.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> some time-series models forecast using known future conditions —
/// "next month is a holiday", "the promotion runs for N days". Those are <i>exogenous</i>
/// variables. To forecast N steps ahead you pass an N-row matrix whose columns are those
/// future values. The unified <see cref="AiDotNet.Models.Results.AiModelResult{T, TInput, TOutput}.Predict"/>
/// front routes such models here, so callers use one <c>Predict</c> entry point for every
/// time-series model class (univariate, exogenous, multivariate).
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public interface IExogenousForecastModel<T>
{
    /// <summary>
    /// Forecasts future target values from the future exogenous regressors. The horizon is
    /// <paramref name="futureExogenous"/>.Rows; its columns are the exogenous variables for
    /// each future step.
    /// </summary>
    /// <param name="futureExogenous">An [horizon x exogenousCount] matrix of future regressors.</param>
    /// <returns>A length-<c>horizon</c> vector of forecasted target values.</returns>
    Vector<T> ForecastWithExogenous(Matrix<T> futureExogenous);
}
