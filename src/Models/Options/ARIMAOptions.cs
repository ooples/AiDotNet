namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model.
/// </summary>
/// <typeparam name="T">The data type of the time series values.</typeparam>
/// <remarks>
/// <para>
/// ARIMA is a statistical model used for analyzing and forecasting time series data. It combines three components:
/// AutoRegressive (AR), Integrated (I), and Moving Average (MA) to model time series data that exhibits non-stationarity.
/// </para>
/// <para><b>For Beginners:</b> ARIMA is a popular method for predicting future values in a time series (data collected 
/// over time, like daily temperatures or monthly sales). It works by combining three techniques: looking at how past values 
/// influence future ones (AutoRegressive), removing trends by taking differences between consecutive values (Integrated), 
/// and accounting for the impact of past prediction errors (Moving Average). Think of it like predicting tomorrow's weather 
/// by considering today's weather, the recent trend of warming or cooling, and how accurate previous forecasts have been.</para>
/// </remarks>
public class ARIMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the AutoRegressive (AR) component.
    /// </summary>
    /// <value>The AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The AR order (P) specifies how many previous time steps are used to predict the current value.
    /// For example, P=1 means the model uses only the immediately preceding value.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many past data points the model looks at to make predictions.
    /// With the default value of 1, the model only considers what happened in the previous time period. If you set it to 2,
    /// it would look at the last two time periods, and so on. Think of it like predicting tomorrow's temperature based on
    /// today's temperature (P=1) versus considering both today and yesterday (P=2). Higher values can capture more complex
    /// patterns but might make the model unnecessarily complicated for simple data.</para>
    /// </remarks>
    public int P { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of differencing (Integration).
    /// </summary>
    /// <value>The differencing order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The differencing order (D) specifies how many times the data is differenced to achieve stationarity.
    /// Differencing helps remove trends and seasonal patterns from the data.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how many times the model subtracts consecutive values to remove trends.
    /// With the default value of 1, the model works with the changes between consecutive values rather than the raw values themselves.
    /// For example, instead of using temperatures like [70°F, 72°F, 75°F], it would use the differences [2°F, 3°F].
    /// This helps the model focus on how values are changing rather than their absolute values, which is useful when data has
    /// an upward or downward trend. Think of it like focusing on how much warmer or cooler it gets each day rather than the
    /// actual temperature.</para>
    /// </remarks>
    public int D { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the Moving Average (MA) component.
    /// </summary>
    /// <value>The MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order (Q) specifies how many previous forecast errors are used in the model.
    /// For example, Q=1 means the model incorporates the error from the previous prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many past prediction errors the model considers.
    /// With the default value of 1, the model adjusts based on how wrong its most recent prediction was.
    /// Think of it like a weather forecaster who not only looks at yesterday's weather but also considers how wrong
    /// their previous forecast was, learning from their mistakes. If yesterday's prediction was too high, they might
    /// adjust today's prediction downward a bit to compensate.</para>
    /// </remarks>
    public int Q { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal AutoRegressive order for seasonal ARIMA models.
    /// </summary>
    /// <value>The seasonal AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The seasonal AR order specifies how many previous seasonal time steps are used to predict the current value.
    /// This captures patterns that repeat at regular intervals (e.g., yearly, monthly).
    /// </para>
    /// <para><b>For Beginners:</b> This is similar to the regular P value, but for seasonal patterns.
    /// It determines how many past seasons influence the current prediction. With the default value of 1,
    /// the model considers what happened in the same period last season. For example, when predicting December sales,
    /// it would consider last December's sales. This helps capture yearly patterns like holiday shopping spikes or
    /// seasonal temperature changes.</para>
    /// </remarks>
    public int SeasonalP { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal differencing order for seasonal ARIMA models.
    /// </summary>
    /// <value>The seasonal differencing order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The seasonal differencing order specifies how many times the data is differenced at the seasonal level
    /// to remove seasonal trends.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the regular D value, but for seasonal differences.
    /// With the default value of 1, the model compares each value to the same period in the previous season.
    /// For example, it might look at the difference between this December's sales and last December's sales.
    /// This helps remove predictable seasonal patterns, allowing the model to focus on other factors affecting the data.</para>
    /// </remarks>
    public int SeasonalD { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal Moving Average order for seasonal ARIMA models.
    /// </summary>
    /// <value>The seasonal MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The seasonal MA order specifies how many previous seasonal forecast errors are used in the model.
    /// This helps account for errors that follow seasonal patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This is similar to the regular Q value, but for seasonal prediction errors.
    /// With the default value of 1, the model considers how wrong its prediction was for the same period last season.
    /// For example, if last December's sales forecast was too high, the model might adjust this December's forecast downward.
    /// This helps the model learn from seasonal forecasting mistakes.</para>
    /// </remarks>
    public int SeasonalQ { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate for the optimization algorithm.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls how quickly the model parameters are updated during training.
    /// A smaller value leads to slower but potentially more precise convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the model adjusts its predictions based on errors.
    /// The default value of 0.01 means the model makes small, cautious adjustments. Think of it like turning a steering
    /// wheel - a small learning rate makes tiny adjustments (good for fine-tuning), while a larger value makes bigger
    /// adjustments (which might help learn faster but could overshoot the optimal solution). For most cases, the default
    /// small value works well because it helps the model find more accurate predictions, even if it takes a bit longer.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits how long the model will train before stopping, preventing excessive computation time.
    /// The algorithm will stop either when it converges or when it reaches this number of iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This is simply a safety limit on how long the model will try to improve itself.
    /// The default value of 1000 means the model will make at most 1000 attempts to refine its predictions before stopping.
    /// Think of it like telling someone they can have up to 1000 tries to solve a puzzle - they might solve it sooner,
    /// but they won't keep trying forever if they're struggling to make progress.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.00001 (1e-5).</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the improvement between iterations is smaller than this tolerance value,
    /// indicating that the model has converged to a solution.
    /// </para>
    /// <para><b>For Beginners:</b> This determines when the model decides it's "good enough" and stops trying to improve.
    /// The default value of 0.00001 means the model will stop when additional training produces very tiny improvements
    /// (less than one hundred-thousandth). Think of it like painting a wall - at some point, adding another coat of paint
    /// makes such a small difference that it's not worth the effort. This parameter defines that "barely noticeable difference"
    /// threshold.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to include an intercept (constant term) in the model.
    /// </summary>
    /// <value>True to include an intercept, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When true, the model includes a constant term that represents the base level of the time series
    /// when all other factors are zero.
    /// </para>
    /// <para><b>For Beginners:</b> This determines whether the model includes a "starting point" or baseline value.
    /// With the default value of true, the model can adjust this baseline up or down to better fit the data.
    /// Think of it like setting a thermostat - the intercept is like the base temperature setting, and the other
    /// parameters make adjustments from there. Having an intercept usually improves predictions, especially when
    /// your data doesn't naturally center around zero.</para>
    /// </remarks>
    public bool FitIntercept { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable anomaly detection during and after training.
    /// </summary>
    /// <value>True to enable anomaly detection, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the model will compute prediction residuals during training and use them
    /// to establish an anomaly threshold based on statistical properties of the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> Setting this to true tells the model to not just predict values,
    /// but also identify unusual data points (anomalies). An anomaly is a value that deviates
    /// significantly from what the model would expect based on the historical patterns it has learned.
    /// For example, a sudden spike or drop that doesn't fit the usual pattern would be flagged as an anomaly.
    /// </para>
    /// </remarks>
    public bool EnableAnomalyDetection { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of standard deviations from the mean residual to use as the anomaly threshold.
    /// </summary>
    /// <value>The anomaly threshold in standard deviations, defaulting to 3.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how sensitive the anomaly detection is. A value of 3.0 means that
    /// data points with residuals more than 3 standard deviations from the mean are flagged as anomalies.
    /// Lower values (like 2.0) will flag more points as anomalies, while higher values (like 4.0) will
    /// only flag the most extreme deviations.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how unusual a value needs to be before it's flagged as an anomaly.
    /// Think of it like a "weirdness threshold":
    /// - A value of 2.0 is lenient - flags anything moderately unusual (about 5% of normal values might be flagged)
    /// - A value of 3.0 (default) is standard - flags clearly unusual values (about 0.3% of normal values)
    /// - A value of 4.0 is strict - only flags extreme outliers (about 0.01% of normal values)
    ///
    /// If you're seeing too many false positives (normal values flagged as anomalies), increase this value.
    /// If you're missing real anomalies, decrease this value.
    /// </para>
    /// </remarks>
    public double AnomalyThresholdSigma { get; set; } = 3.0;
}
