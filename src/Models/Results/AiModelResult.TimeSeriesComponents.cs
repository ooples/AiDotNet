using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

namespace AiDotNet.Models.Results;

/// <summary>
/// What a fitted time-series model found in the series, beyond its forecast: the decomposition it
/// produced, the frequencies it identified, and the points it considers anomalous.
/// </summary>
/// <remarks>
/// <para>
/// Forecasting already runs through <see cref="Predict(Vector{T}, int)"/>. These are the other reasons to
/// fit a time-series model — you decompose a series to *see* the trend and the seasonality, and that
/// output is the deliverable, not an intermediate step toward a prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b> A sales series is usually three things added together: a long-run direction
/// (trend), a repeating yearly or weekly pattern (seasonal), and what is left over (residual). Splitting
/// them apart tells you whether a good month was real growth or just December.
/// </para>
/// </remarks>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the trend component the model separated out of the series.
    /// </summary>
    /// <returns>The trend, one value per point in the fitted series.</returns>
    /// <exception cref="NotSupportedException">The built model does not decompose a series.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The trend is the slow-moving direction underneath the noise — whether sales
    /// are really rising once you ignore the seasonal ups and downs.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var series = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new STLDecomposition&lt;double&gt;(new STLDecompositionOptions&lt;double&gt;()))
    ///     .Build(features, series);
    ///
    /// var trend = result.GetTrend();
    /// </code>
    /// </example>
    public Vector<T> GetTrend()
    {
        if (EnsureModel is STLDecomposition<T> stl)
        {
            return stl.GetTrend();
        }

        throw new NotSupportedException(
            $"GetTrend requires a decomposition model (STLDecomposition<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Gets the seasonal component the model separated out of the series.
    /// </summary>
    /// <returns>The seasonal component, one value per point in the fitted series.</returns>
    /// <exception cref="NotSupportedException">The built model does not decompose a series.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The repeating pattern — the December spike, the Monday dip — with the trend
    /// and the noise taken out.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var series = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new STLDecomposition&lt;double&gt;(new STLDecompositionOptions&lt;double&gt;()))
    ///     .Build(features, series);
    ///
    /// var seasonal = result.GetSeasonal();
    /// </code>
    /// </example>
    public Vector<T> GetSeasonal()
    {
        if (EnsureModel is STLDecomposition<T> stl)
        {
            return stl.GetSeasonal();
        }

        throw new NotSupportedException(
            $"GetSeasonal requires a decomposition model (STLDecomposition<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Gets the frequencies at which the spectral model evaluated the series.
    /// </summary>
    /// <returns>The frequency grid.</returns>
    /// <exception cref="NotSupportedException">The built model is not a spectral analysis.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frequency is "how often does this repeat". Paired with
    /// <see cref="GetPeriodogram"/>, it tells you which repeat lengths carry the most of the signal — a
    /// spike at 1/7 in daily data means a weekly cycle.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var signalMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var signalVector = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new SpectralAnalysisModel&lt;double&gt;(new SpectralAnalysisOptions&lt;double&gt;()))
    ///     .Build(signalMatrix, signalVector);
    ///
    /// var frequencies = result.GetFrequencies();
    /// </code>
    /// </example>
    public Vector<T> GetFrequencies()
    {
        if (EnsureModel is SpectralAnalysisModel<T> spectral)
        {
            return spectral.GetFrequencies();
        }

        throw new NotSupportedException(
            $"GetFrequencies requires a spectral model (SpectralAnalysisModel<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Gets the periodogram: how much of the series' variance sits at each frequency.
    /// </summary>
    /// <returns>The power at each frequency returned by <see cref="GetFrequencies"/>.</returns>
    /// <exception cref="NotSupportedException">The built model is not a spectral analysis.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Read it alongside <see cref="GetFrequencies"/>: the tall peaks are the
    /// cycles that actually drive your series.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var signalMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var signalVector = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new SpectralAnalysisModel&lt;double&gt;(new SpectralAnalysisOptions&lt;double&gt;()))
    ///     .Build(signalMatrix, signalVector);
    ///
    /// var periodogram = result.GetPeriodogram();
    /// </code>
    /// </example>
    public Vector<T> GetPeriodogram()
    {
        if (EnsureModel is SpectralAnalysisModel<T> spectral)
        {
            return spectral.GetPeriodogram();
        }

        throw new NotSupportedException(
            $"GetPeriodogram requires a spectral model (SpectralAnalysisModel<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Scores how unusual each point of a series is, given what the fitted model expected.
    /// </summary>
    /// <param name="timeSeries">The series to score.</param>
    /// <returns>One anomaly score per point; larger means more surprising.</returns>
    /// <exception cref="NotSupportedException">The built model does not score anomalies.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model learned what the series normally does. This asks, point by point,
    /// "how far off was reality from that?" — which is how you find the outage, the fraud, or the typo.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var trainingLabels = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new ARIMAModel&lt;double&gt;(new ARIMAOptions&lt;double&gt;()))
    ///     .Build(trainingMatrix, trainingLabels);
    ///
    /// var timeSeries = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    /// var anomalyScores = result.ComputeAnomalyScores(timeSeries);
    /// </code>
    /// </example>
    public Vector<T> ComputeAnomalyScores(Vector<T> timeSeries)
    {
        if (EnsureModel is ARIMAModel<T> arima)
        {
            return arima.ComputeAnomalyScores(timeSeries);
        }

        throw new NotSupportedException(
            $"ComputeAnomalyScores requires an ARIMA model (ARIMAModel<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Forecasts forward from a history you supply, optionally with exogenous drivers.
    /// </summary>
    /// <param name="history">The observed series to extend.</param>
    /// <param name="horizon">How many steps ahead to forecast.</param>
    /// <param name="exogenousVariables">Future values of any external drivers, one row per step ahead.</param>
    /// <returns>The forecast, one value per step.</returns>
    /// <exception cref="NotSupportedException">The built model does not forecast from a supplied history.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this rather than <see cref="Predict(Vector{T}, int)"/> when your model
    /// also depends on things outside the series itself — a price, a promotion, the weather.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var trainingLabels = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new BayesianStructuralTimeSeriesModel&lt;double&gt;(new BayesianStructuralTimeSeriesOptions&lt;double&gt;()))
    ///     .Build(trainingMatrix, trainingLabels);
    ///
    /// var history = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    /// var forecast = result.Forecast(history, horizon: 12);
    /// </code>
    /// </example>
    public Vector<T> Forecast(Vector<T> history, int horizon, Matrix<T>? exogenousVariables = null)
    {
        if (EnsureModel is BayesianStructuralTimeSeriesModel<T> bsts)
        {
            return bsts.Forecast(history, horizon, exogenousVariables);
        }

        throw new NotSupportedException(
            $"Forecast(history, horizon) requires a structural time-series model " +
            $"(BayesianStructuralTimeSeriesModel<T>); the built model is {EnsureModel.GetType().Name}. " +
            $"For ordinary sequence models use Predict(lookback, horizon).");
    }

    /// <summary>
    /// Forecasts forward from the series the model was fitted on.
    /// </summary>
    /// <param name="horizon">How many steps ahead to forecast.</param>
    /// <param name="startIndex">Where in the fitted series to forecast from; -1 continues from the end.</param>
    /// <returns>The forecast, one value per step.</returns>
    /// <exception cref="NotSupportedException">The built model does not forecast from its own history.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model already saw the series when it was built, so it does not need it
    /// again — just say how far ahead you want to look.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
    /// var trainingLabels = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new UnobservedComponentsModel&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(
    ///         new UnobservedComponentsOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()))
    ///     .Build(trainingMatrix, trainingLabels);
    ///
    /// var forecast = result.Forecast(horizon: 24);
    /// </code>
    /// </example>
    public Vector<T> Forecast(int horizon, int startIndex = -1)
    {
        if (EnsureModel is UnobservedComponentsModel<T, TInput, TOutput> ucm)
        {
            return ucm.Forecast(horizon, startIndex);
        }

        throw new NotSupportedException(
            $"Forecast(horizon) requires an unobserved-components model; the built model is " +
            $"{EnsureModel.GetType().Name}. For ordinary sequence models use Predict(lookback, horizon).");
    }
}
