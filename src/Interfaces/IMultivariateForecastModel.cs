namespace AiDotNet.Interfaces;

/// <summary>
/// Marks a <b>multivariate</b> time-series model that forecasts several interrelated series
/// at once (e.g. VAR, VARMA). A multivariate forecast is naturally a
/// [horizon x <see cref="VariableCount"/>] matrix.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> a multivariate model forecasts several series together because they
/// influence each other (GDP and unemployment, temperature and energy use). Forecasting N
/// steps produces N rows, one column per series. The unified
/// <see cref="AiDotNet.Models.Results.AiModelResult{T, TInput, TOutput}.Predict"/> front routes
/// such models here and returns the result through the common <c>Vector&lt;T&gt;</c> output
/// (the matrix flattened row-major, with the [horizon, variables] shape recorded on the
/// result so it can be reshaped back to a matrix for the caller — no layout knowledge required).
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public interface IMultivariateForecastModel<T>
{
    /// <summary>The number of interrelated series (variables) the model forecasts.</summary>
    int VariableCount { get; }

    /// <summary>
    /// Forecasts <paramref name="horizon"/> steps for all variables, seeded from the
    /// multivariate <paramref name="history"/> the model was trained on.
    /// </summary>
    /// <param name="history">An [observations x <see cref="VariableCount"/>] history matrix.</param>
    /// <param name="horizon">The number of future steps to forecast.</param>
    /// <returns>A [horizon x <see cref="VariableCount"/>] forecast matrix.</returns>
    Matrix<T> ForecastMultivariate(Matrix<T> history, int horizon);
}
