namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for time series regression models, which analyze data collected over time
/// to identify patterns and make predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Time series regression extends traditional regression analysis to account for the temporal nature of data,
/// where observations are collected sequentially over time. These models can capture trends, seasonal patterns,
/// and the effects of past values on current and future values.
/// </para>
/// <para><b>For Beginners:</b> Time series regression helps you analyze and predict data that changes over time,
/// like stock prices, weather patterns, or monthly sales figures. Unlike regular regression that just looks for
/// relationships between variables, time series regression also considers when things happened. It can detect
/// patterns like "sales always increase in December" or "temperature today is related to temperature yesterday."
/// This class lets you configure how the model analyzes these time-based patterns.</para>
/// </remarks>
public class TimeSeriesRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the lag order, which determines how many previous time steps are used as predictors.
    /// </summary>
    /// <value>The lag order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The lag order specifies how many previous observations are included as explanatory variables in the model.
    /// For example, a lag order of 2 means that values from both t-1 and t-2 time steps are used to predict the value at time t.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how far back in time the model looks when making predictions.
    /// With the default value of 1, the model considers only the previous time period (like yesterday's value when
    /// predicting today's). If you set it to 3, the model would look at the three previous time periods
    /// (like the last three days' values). Higher values help capture longer-term patterns but require more data
    /// and can make the model more complex.</para>
    /// </remarks>
    public int LagOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to include a trend component in the model.
    /// </summary>
    /// <value>True to include a trend component, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the model will include a linear trend term to capture consistent upward or downward movement in the data.
    /// This helps the model account for long-term directional changes that aren't explained by other variables.
    /// </para>
    /// <para><b>For Beginners:</b> This determines whether the model should look for overall upward or downward
    /// movement in your data over time. With the default value of true, the model will try to identify if your data
    /// is generally increasing or decreasing (like a growing customer base or declining costs). If your data doesn't
    /// have a clear direction over time, you might set this to false to simplify the model.</para>
    /// </remarks>
    public bool IncludeTrend { get; set; } = true;

    /// <summary>
    /// Gets or sets the seasonal period of the time series data.
    /// </summary>
    /// <value>The seasonal period, defaulting to 0 (no seasonality).</value>
    /// <remarks>
    /// <para>
    /// The seasonal period represents how many time steps make up one complete cycle of seasonal pattern.
    /// For example, 12 for monthly data with yearly seasonality, 7 for daily data with weekly seasonality.
    /// A value of 0 indicates that no seasonal component should be included in the model.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the model if your data has regular patterns that repeat at fixed intervals.
    /// The default value of 0 means "don't look for seasonal patterns." If you're analyzing monthly data and expect
    /// yearly patterns (like retail sales peaking every December), you would set this to 12. For daily data with
    /// weekly patterns, you'd use 7. For quarterly data with yearly patterns, you'd use 4. This helps the model
    /// recognize and predict these recurring cycles.</para>
    /// </remarks>
    public int SeasonalPeriod { get; set; } = 0; // 0 means no seasonality

    /// <summary>
    /// Gets or sets whether to apply autocorrelation correction to the model.
    /// </summary>
    /// <value>True to apply autocorrelation correction, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// Autocorrelation occurs when error terms in a time series are correlated across observations, violating
    /// standard regression assumptions. When this option is enabled, the model applies methods to correct for
    /// this correlation, improving the accuracy of coefficient estimates and predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This determines whether the model should adjust for the fact that errors in
    /// time series data tend to be related to each other. With the default value of true, the model will make
    /// these adjustments, which usually improves accuracy. For example, if the model consistently underestimates
    /// values for several days in a row, this correction helps it recognize and fix that pattern. You should
    /// generally leave this enabled unless you have a specific reason to disable it.</para>
    /// </remarks>
    public bool AutocorrelationCorrection { get; set; } = true;

    /// <summary>
    /// Gets or sets the specific type of time series model to use.
    /// </summary>
    /// <value>The time series model type, defaulting to ARIMA.</value>
    /// <remarks>
    /// <para>
    /// Different time series model types have different capabilities and are suited to different types of data.
    /// ARIMA (AutoRegressive Integrated Moving Average) is a flexible and widely-used approach that can handle
    /// many common time series patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This selects which specific algorithm the model uses to analyze your time series data.
    /// The default is ARIMA, which is a popular and versatile method that works well for many types of time data.
    /// Other options might include methods like exponential smoothing (good for data with clear trends and seasonality)
    /// or VAR (Vector AutoRegression, good for analyzing relationships between multiple time series).
    /// Unless you're familiar with the different methods, starting with ARIMA is usually a good choice.</para>
    /// </remarks>
    public TimeSeriesModelType ModelType { get; set; } = TimeSeriesModelType.ARIMA;

    /// <summary>
    /// Gets or sets the loss function used for gradient computation and model training.
    /// </summary>
    /// <value>The loss function, defaulting to null (which will use MeanSquaredErrorLoss).</value>
    /// <remarks>
    /// <para>
    /// The loss function determines how the model measures prediction errors during training.
    /// Different loss functions are appropriate for different types of problems and data characteristics.
    /// If null, the model will use Mean Squared Error (MSE) as the default loss function.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how the model measures its mistakes during training.
    /// The default (null) uses Mean Squared Error, which works well for most time series forecasting tasks.
    /// You can provide a custom loss function if you have specific requirements, such as:
    /// - MeanAbsoluteErrorLoss for robustness to outliers
    /// - HuberLoss for a balance between MSE and MAE
    /// - Custom loss functions for domain-specific error metrics
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; } = null;
}
