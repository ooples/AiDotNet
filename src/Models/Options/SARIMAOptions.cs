namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Seasonal Autoregressive Integrated Moving Average (SARIMA) models,
/// which extend ARIMA models to incorporate seasonal components in time series data.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the time series model.</typeparam>
/// <remarks>
/// <para>
/// SARIMA (Seasonal Autoregressive Integrated Moving Average) is a sophisticated time series forecasting model 
/// that extends the ARIMA (Autoregressive Integrated Moving Average) framework to handle seasonal patterns in data. 
/// It is particularly useful for time series that exhibit both trend and seasonality, such as monthly sales data, 
/// quarterly economic indicators, or daily web traffic with weekly patterns. The model is denoted as 
/// SARIMA(p,d,q)(P,D,Q)m, where p, d, q are the non-seasonal parameters (autoregressive order, differencing order, 
/// and moving average order), P, D, Q are the corresponding seasonal parameters, and m is the seasonal period. 
/// This class provides configuration options for all these parameters, allowing fine-tuning of the SARIMA model 
/// to best capture the specific characteristics of the time series being analyzed.
/// </para>
/// <para><b>For Beginners:</b> SARIMA helps predict future values in time series data that has seasonal patterns.
/// 
/// Time series data shows how values change over time, like:
/// - Monthly sales figures
/// - Daily temperature readings
/// - Quarterly company earnings
/// 
/// Many time series have seasonal patterns:
/// - Retail sales spike during holidays
/// - Ice cream consumption increases in summer
/// - Energy usage follows daily and yearly cycles
/// 
/// SARIMA is designed to capture both:
/// - The overall trend in your data
/// - The repeating seasonal patterns
/// 
/// It does this by combining:
/// - Regular ARIMA components that handle short-term patterns and trends
/// - Seasonal components that handle repeating patterns at fixed intervals
/// 
/// This class lets you configure exactly how the SARIMA model analyzes your time series data,
/// including how many past values to consider and how to handle seasonality.
/// </para>
/// </remarks>
public class SARIMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the autoregressive order (p) of the non-seasonal component.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The autoregressive order (p) determines how many lagged observations are included in the model. It 
    /// represents the number of past time points that directly influence the current value. A higher p value 
    /// allows the model to capture more complex autocorrelation patterns but increases the risk of overfitting 
    /// and computational complexity. The default value of 1 means that the model includes the influence of the 
    /// immediately preceding observation, which is sufficient for many time series with simple autocorrelation 
    /// structures. For time series with more complex dependencies on past values, a higher p value might be 
    /// appropriate. Model selection techniques like AIC (Akaike Information Criterion) or BIC (Bayesian Information 
    /// Criterion) are often used to determine the optimal p value.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many previous time points directly influence the current prediction.
    /// 
    /// The P value determines:
    /// - How many past observations the model uses to predict the current value
    /// - How directly the model captures the relationship between current and past values
    /// 
    /// The default value of 1 means:
    /// - The model considers the immediate previous value (t-1) when predicting the current value (t)
    /// - This works well for many simple time series
    /// 
    /// Think of it like this:
    /// - P=1: Today's temperature depends on yesterday's temperature
    /// - P=2: Today's temperature depends on both yesterday's and the day before's temperatures
    /// - P=3: Extends this pattern to three previous days
    /// 
    /// When to adjust this value:
    /// - Increase it (to 2, 3, etc.) when you believe values depend on multiple previous time points
    /// - Keep it low (0 or 1) for simpler patterns or to avoid overfitting with limited data
    /// 
    /// For example, in stock price prediction, a P value of 2 or 3 might capture short-term momentum
    /// effects where recent price movements influence current price changes.
    /// </para>
    /// </remarks>
    public int P { get; set; } = 1;

    /// <summary>
    /// Gets or sets the differencing order (d) of the non-seasonal component.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// The differencing order (d) specifies how many times the time series is differenced to achieve stationarity. 
    /// Differencing involves computing the differences between consecutive observations and is used to remove trends 
    /// from the data. A value of 0 means no differencing is applied, which is appropriate for already stationary 
    /// series. A value of 1 applies first-order differencing, which can remove linear trends. A value of 2 applies 
    /// second-order differencing, which can remove quadratic trends. Higher values are rarely needed in practice. 
    /// The default value of 0 assumes that the time series is already stationary or that any non-stationarity will 
    /// be addressed by other components of the model. Statistical tests like the Augmented Dickey-Fuller test can 
    /// help determine the appropriate differencing order.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the model handles trends in your data.
    /// 
    /// Differencing means taking the difference between consecutive values:
    /// - It helps remove trends and make the data more stable (stationary)
    /// - Each level of differencing removes a more complex trend
    /// 
    /// The default value of 0 means:
    /// - No differencing is applied
    /// - The model assumes your data doesn't have strong trends or that other components will handle them
    /// 
    /// Different values mean:
    /// - D=0: No trend removal (use when data has no trend or when you want to preserve the trend)
    /// - D=1: Removes linear trends (like steady growth or decline)
    /// - D=2: Removes quadratic trends (like accelerating growth or decline)
    /// 
    /// When to adjust this value:
    /// - Increase to 1 when your data shows clear upward or downward trends
    /// - Increase to 2 when trends are changing (accelerating or decelerating)
    /// - Keep at 0 when data fluctuates around a constant level
    /// 
    /// For example, with steadily increasing monthly sales data, setting D=1 would help the model
    /// focus on the patterns in the growth rather than the absolute values.
    /// </para>
    /// </remarks>
    public int D { get; set; } = 0;

    /// <summary>
    /// Gets or sets the moving average order (q) of the non-seasonal component.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The moving average order (q) determines how many lagged forecast errors are included in the model. It 
    /// represents the number of past random shocks (unexpected events or innovations) that influence the current 
    /// value. A higher q value allows the model to capture more complex patterns in the residuals but increases 
    /// the risk of overfitting and computational complexity. The default value of 1 means that the model includes 
    /// the influence of the immediately preceding forecast error, which is sufficient for many time series with 
    /// simple error structures. For time series with more complex error patterns, a higher q value might be 
    /// appropriate. As with the autoregressive order, model selection techniques like AIC or BIC are often used 
    /// to determine the optimal q value.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the model handles unexpected events or shocks in your data.
    /// 
    /// The Q value determines:
    /// - How many past "surprises" or errors influence the current prediction
    /// - How the model accounts for unexpected events that caused previous predictions to be off
    /// 
    /// The default value of 1 means:
    /// - The model considers the most recent prediction error when making new predictions
    /// - This helps adjust for recent unexpected events
    /// 
    /// Think of it like this:
    /// - Q=1: If yesterday's weather was unexpectedly rainy, today's forecast adjusts for that surprise
    /// - Q=2: Adjusts for surprises from both yesterday and the day before
    /// - Q=3: Extends this pattern to three previous days
    /// 
    /// When to adjust this value:
    /// - Increase it when unexpected events have lingering effects over multiple time periods
    /// - Keep it low when shocks to the system are quickly absorbed
    /// 
    /// For example, in retail sales, a Q value of 2 might help account for how unexpected events
    /// (like a promotion or supply shortage) affect sales for a couple of periods afterward.
    /// </para>
    /// </remarks>
    public int Q { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal autoregressive order (P) of the model.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The seasonal autoregressive order (P) determines how many seasonally lagged observations are included in 
    /// the model. It represents the number of past seasonal time points that directly influence the current value. 
    /// For example, with monthly data and a seasonal period of 12, a P value of 1 means that the value from the 
    /// same month in the previous year influences the current value. A higher P value allows the model to capture 
    /// more complex seasonal autocorrelation patterns but increases the risk of overfitting and computational 
    /// complexity. The default value of 1 is sufficient for many seasonal time series with simple seasonal 
    /// autocorrelation structures. For time series with more complex seasonal dependencies, a higher P value 
    /// might be appropriate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how previous seasons influence the current prediction.
    /// 
    /// The SeasonalP value determines:
    /// - How many previous seasonal periods directly influence the current value
    /// - How the model captures year-over-year (or season-over-season) patterns
    /// 
    /// The default value of 1 means:
    /// - The model considers the same time point from the previous season
    /// - For monthly data with 12-month seasonality, January 2023 is influenced by January 2022
    /// 
    /// Think of it like this:
    /// - SeasonalP=1: This month's sales depend on the same month last year
    /// - SeasonalP=2: This month's sales depend on the same month from both last year and two years ago
    /// 
    /// When to adjust this value:
    /// - Increase it when you believe multiple previous seasons influence the current season
    /// - Keep it at 1 for most seasonal patterns, as one previous season is often sufficient
    /// 
    /// For example, in tourism data, a SeasonalP of 1 would capture how summer tourism this year
    /// is related to summer tourism last year.
    /// </para>
    /// </remarks>
    public int SeasonalP { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal differencing order (D) of the model.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// The seasonal differencing order (D) specifies how many times the time series is seasonally differenced to 
    /// achieve seasonal stationarity. Seasonal differencing involves computing the differences between observations 
    /// separated by the seasonal period and is used to remove seasonal trends from the data. A value of 0 means no 
    /// seasonal differencing is applied, which is appropriate for series without seasonal trends. A value of 1 
    /// applies first-order seasonal differencing, which can remove seasonal patterns that change linearly over time. 
    /// Higher values are rarely needed in practice. The default value of 0 assumes that any seasonal patterns in the 
    /// time series are stable over time or will be addressed by other components of the model. Statistical tests and 
    /// visual inspection of seasonal plots can help determine the appropriate seasonal differencing order.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the model handles changing seasonal patterns.
    /// 
    /// Seasonal differencing compares values across seasons:
    /// - It helps remove changing seasonal patterns
    /// - It's like comparing "this January vs. last January" instead of "January vs. December"
    /// 
    /// The default value of 0 means:
    /// - No seasonal differencing is applied
    /// - The model assumes seasonal patterns are stable over time
    /// 
    /// Different values mean:
    /// - SeasonalD=0: No seasonal trend removal (seasons have consistent patterns)
    /// - SeasonalD=1: Removes changing seasonal patterns (when seasonal effects grow or shrink over time)
    /// 
    /// When to adjust this value:
    /// - Increase to 1 when seasonal patterns are getting stronger or weaker over time
    /// - Keep at 0 when seasonal patterns are consistent year after year
    /// 
    /// For example, if holiday sales spikes are getting larger each year, setting SeasonalD=1 would
    /// help the model account for this growing seasonal effect.
    /// </para>
    /// </remarks>
    public int SeasonalD { get; set; } = 0;

    /// <summary>
    /// Gets or sets the seasonal moving average order (Q) of the model.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The seasonal moving average order (Q) determines how many seasonally lagged forecast errors are included in 
    /// the model. It represents the number of past seasonal random shocks that influence the current value. For 
    /// example, with monthly data and a seasonal period of 12, a Q value of 1 means that the forecast error from 
    /// the same month in the previous year influences the current value. A higher Q value allows the model to 
    /// capture more complex seasonal error patterns but increases the risk of overfitting and computational 
    /// complexity. The default value of 1 is sufficient for many seasonal time series with simple seasonal error 
    /// structures. For time series with more complex seasonal error patterns, a higher Q value might be appropriate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how unexpected seasonal events influence the current prediction.
    /// 
    /// The SeasonalQ value determines:
    /// - How many past seasonal "surprises" or errors influence the current prediction
    /// - How the model accounts for unexpected events that happened in previous seasons
    /// 
    /// The default value of 1 means:
    /// - The model considers the prediction error from the same time in the previous season
    /// - For monthly data, an unexpected spike last December affects this December's prediction
    /// 
    /// Think of it like this:
    /// - SeasonalQ=1: If last Christmas had unexpectedly high sales, this Christmas's forecast adjusts for that
    /// - SeasonalQ=2: Adjusts for surprises from both last Christmas and the Christmas before that
    /// 
    /// When to adjust this value:
    /// - Increase it when seasonal shocks have effects that persist across multiple years
    /// - Keep it at 1 for most cases, as one seasonal period is often sufficient
    /// 
    /// For example, in energy consumption data, a SeasonalQ of 1 would help account for how an
    /// unusually cold winter last year might inform predictions for this winter.
    /// </para>
    /// </remarks>
    public int SeasonalQ { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of time points in one seasonal cycle.
    /// </summary>
    /// <value>A positive integer, defaulting to 12 (monthly seasonality).</value>
    /// <remarks>
    /// <para>
    /// The seasonal period defines the number of time points that make up one complete seasonal cycle in the data. 
    /// It represents the periodicity of the seasonal pattern. Common values include 12 for monthly data with yearly 
    /// seasonality, 4 for quarterly data with yearly seasonality, 7 for daily data with weekly seasonality, or 24 
    /// for hourly data with daily seasonality. The default value of 12 is appropriate for monthly data with yearly 
    /// seasonality, which is common in many business and economic applications. This property overrides the 
    /// SeasonalPeriod property inherited from the base class to provide a more appropriate default value for 
    /// SARIMA models. The correct specification of the seasonal period is crucial for the model to properly 
    /// capture the seasonal patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines how long one complete seasonal cycle is in your data.
    /// 
    /// The SeasonalPeriod tells the model how many time points make up a complete cycle:
    /// - For monthly data: 12 (default) means a yearly seasonal pattern
    /// - For quarterly data: 4 would mean a yearly seasonal pattern
    /// - For daily data: 7 would mean a weekly seasonal pattern
    /// - For hourly data: 24 would mean a daily seasonal pattern
    /// 
    /// This value is critical because it tells the model exactly how far back to look for seasonal patterns:
    /// - With monthly data and SeasonalPeriod=12, January is compared with previous Januaries
    /// - With daily data and SeasonalPeriod=7, Mondays are compared with previous Mondays
    /// 
    /// When to adjust this value:
    /// - Change it to match the natural cycle in your data
    /// - Common values: 12 (monthly), 4 (quarterly), 7 (daily with weekly pattern), 24 (hourly)
    /// 
    /// For example, if analyzing weekly sales data over multiple years, you might set this to 52
    /// to capture yearly seasonality (52 weeks in a year).
    /// </para>
    /// </remarks>
    public new int SeasonalPeriod { get; set; } = 12;  // Default to monthly seasonality

    /// <summary>
    /// Gets or sets the maximum number of iterations for the parameter estimation algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of iterations allowed for the numerical optimization algorithm 
    /// used to estimate the SARIMA model parameters. SARIMA models are typically fitted using maximum likelihood 
    /// estimation, which requires an iterative optimization procedure. This parameter prevents the algorithm from 
    /// running indefinitely in cases where convergence is difficult to achieve. The default value of 1000 is 
    /// sufficient for most applications to achieve convergence. However, for particularly complex models or 
    /// difficult datasets, more iterations may be required. Conversely, for simpler models or when approximate 
    /// solutions are acceptable, fewer iterations may be sufficient. If the algorithm reaches this maximum without 
    /// converging according to the Tolerance criterion, it will return the best solution found so far, possibly 
    /// with a warning.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how long the algorithm will try to find the best model parameters.
    /// 
    /// SARIMA uses an iterative process to find the optimal parameters:
    /// - It starts with initial estimates and gradually improves them
    /// - Each iteration refines the parameters to better fit your data
    /// - The process continues until the improvements become very small or the maximum iterations is reached
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will try up to 1000 rounds of refinement
    /// - This is usually more than enough for most time series
    /// - If it reaches this limit without converging, it returns the best solution found so far
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 2000 or 5000) for complex seasonal patterns or if you get convergence warnings
    /// - Decrease it (e.g., to 500) when you need faster results and approximate solutions are acceptable
    /// 
    /// For example, when fitting a SARIMA model to complex economic data with multiple seasonal
    /// patterns, you might need to increase this value to ensure the algorithm finds the optimal solution.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for the parameter estimation algorithm.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-5 (0.00001).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the convergence criterion for the numerical optimization algorithm used to estimate 
    /// the SARIMA model parameters. The algorithm terminates when the relative change in the log-likelihood function 
    /// or parameter estimates between consecutive iterations is less than this tolerance value. A smaller tolerance 
    /// requires more precise convergence, potentially leading to more accurate parameter estimates but requiring more 
    /// iterations. A larger tolerance allows for earlier termination, potentially saving computational resources at 
    /// the cost of less precise parameter estimates. The default value of 1e-5 (0.00001) provides a good balance 
    /// between precision and computational efficiency for most applications. For high-precision requirements, a 
    /// smaller value might be appropriate, while for exploratory analysis or when computational resources are limited, 
    /// a larger value might be used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the final solution needs to be.
    /// 
    /// During the iterative process, the algorithm keeps refining its solution:
    /// - It compares each new solution to the previous one
    /// - When the improvement becomes smaller than the tolerance value, it stops
    /// - This indicates the solution has stabilized and further iterations won't improve much
    /// 
    /// The Tolerance value (default 0.00001 or 1e-5) means:
    /// - The algorithm will stop when improvements between iterations are very small
    /// - A smaller value requires more precision (and usually more iterations)
    /// - A larger value allows the algorithm to stop earlier with an approximate solution
    /// 
    /// When to adjust this value:
    /// - Decrease it (e.g., to 1e-6 or 1e-7) when you need extremely precise parameter estimates
    /// - Increase it (e.g., to 1e-4 or 1e-3) when you need faster results and can accept slight imprecision
    /// 
    /// For example, in a production forecasting system where small prediction improvements have
    /// significant financial impact, you might use a smaller tolerance to ensure maximum precision.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-5;
}
