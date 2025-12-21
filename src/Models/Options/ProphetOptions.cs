namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Prophet, a procedure for forecasting time series data based on an additive model
/// where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
/// </summary>
/// <remarks>
/// <para>
/// Prophet is a forecasting procedure developed by Facebook (now Meta) that decomposes time series into several components:
/// trend, seasonality, holiday effects, and error. It is designed to handle time series with strong seasonal patterns, 
/// missing values, outliers, and shifts in the trend. Prophet works particularly well with time series that have
/// strong seasonal effects and several seasons of historical data. The model automatically detects changes in trends 
/// by selecting changepoints from the data, while also allowing manual specification of known changepoints. 
/// Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. 
/// It is particularly useful for business and economic time series that are affected by holidays and seasonal patterns.
/// </para>
/// <para><b>For Beginners:</b> Prophet is a powerful tool for predicting future values in time-based data.
///
/// Imagine you're trying to forecast your company's sales for the next few months:
/// - Sales might follow patterns that repeat yearly (holiday shopping seasons)
/// - They might follow weekly patterns (higher on weekends, lower on Mondays)
/// - There might be special days that affect sales (Black Friday, Cyber Monday)
/// - The overall trend might be growing, shrinking, or changing direction occasionally
///
/// What Prophet does:
/// - It breaks down your data into separate pieces (components):
///   - Base trend: The overall direction (growing or declining)
///   - Seasonality: Repeating patterns (yearly, weekly, daily)
///   - Holiday effects: Impacts of special days
///   - Changepoints: Where the trend changes direction
///
/// Think of it like weather forecasting:
/// - It looks at historical patterns
/// - It accounts for seasons and special events
/// - It combines these patterns to predict what will happen next
///
/// The benefit of Prophet is that it automatically handles many complex aspects of time series forecasting
/// that would otherwise require significant expertise. This class lets you configure how Prophet analyzes 
/// and forecasts your time series data.
/// </para>
/// </remarks>
public class ProphetOptions<T, TInput, TOutput> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the initial value for the trend component.
    /// </summary>
    /// <value>The initial trend value, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets the initial value for the trend component at the start of the time series.
    /// It represents the y-intercept (value at time t=0) of the trend line. In most cases, Prophet can 
    /// effectively estimate this value from the data, so the default of 0.0 is generally sufficient.
    /// However, setting a specific initial trend value can be helpful when you have domain knowledge
    /// about the starting point of the series or when working with limited historical data where
    /// you want to inject prior information about the base level of the series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines the starting value of the basic trend in your data.
    ///
    /// The default value of 0.0 means:
    /// - Prophet will estimate the starting point of your trend from the data
    /// - You don't need to specify a starting value manually
    ///
    /// Think of it like the starting point on a map:
    /// - If you're tracking elevation changes on a hike, this would be your elevation at the trailhead
    /// - It doesn't affect the shape of the path, just where it begins on the y-axis
    ///
    /// You might want to set a specific value:
    /// - When you have expert knowledge about the true starting level of your data
    /// - When you have limited historical data and want to provide this prior information
    /// - When you're continuing a forecast from a previously known point
    ///
    /// In most cases, leaving this at the default value works well as Prophet will determine 
    /// an appropriate starting value from your data automatically.
    /// </para>
    /// </remarks>
    public double InitialTrendValue { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the list of periods (in days) for custom seasonality components.
    /// </summary>
    /// <value>A list of seasonal periods, defaulting to an empty list.</value>
    /// <remarks>
    /// <para>
    /// This parameter allows the specification of custom seasonal periods beyond the standard yearly, weekly,
    /// and daily seasonalities. Each value in the list represents a seasonal period in days. For example,
    /// a value of 365.25 would represent annual seasonality, 7 would represent weekly seasonality, and 30.5
    /// would approximate monthly seasonality. Custom seasonalities are particularly useful for domain-specific
    /// cyclical patterns that don't align with calendar periods, such as business quarters, academic semesters,
    /// or specific industry cycles. When custom seasonal periods are specified, Prophet will model these
    /// additional seasonal components alongside any enabled standard seasonalities.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you specify custom repeating patterns in your data beyond the standard yearly, weekly, and daily cycles.
    ///
    /// The default empty list means:
    /// - No custom seasonal patterns will be modeled
    /// - Only the standard seasonalities you enable (yearly, weekly, daily) will be used
    ///
    /// Think of seasonality like repeating patterns:
    /// - Yearly seasonality captures patterns that repeat every 365.25 days
    /// - Weekly seasonality captures patterns that repeat every 7 days
    /// - You might have other patterns that repeat at different intervals
    ///
    /// Examples of custom seasonal periods you might add:
    /// - 91.3 (quarterly patterns in business data)
    /// - 30.5 (monthly patterns)
    /// - 14 (bi-weekly patterns)
    /// - 122 (roughly 4-month seasonal patterns)
    ///
    /// You would add custom seasonalities when:
    /// - Your data has known cycles that don't match the standard calendar periods
    /// - You've observed repeating patterns in your data at specific intervals
    /// - Your industry or domain has specific cyclical behaviors (school semesters, sports seasons, etc.)
    ///
    /// For example: `SeasonalPeriods = new List<int> { 91, 30 }` would add both quarterly and monthly 
    /// seasonality components to your model.
    /// </para>
    /// </remarks>
    public List<int> SeasonalPeriods { get; set; } = new List<int> { };

    /// <summary>
    /// Gets or sets the list of holiday dates that have special effects on the time series.
    /// </summary>
    /// <value>A list of holiday dates, defaulting to an empty list.</value>
    /// <remarks>
    /// <para>
    /// This parameter allows the explicit specification of dates that should be treated as holidays
    /// or special events in the model. Prophet will learn the specific effect of each holiday on the
    /// time series from the historical data. Holidays can cause significant deviations from normal
    /// patterns and explicitly modeling them improves forecast accuracy. The holiday effects are
    /// modeled as additive terms in the forecast, allowing each holiday to have a unique impact.
    /// For recurring holidays with shifting dates (like Easter or Thanksgiving), it's important to
    /// include instances from each historical year as well as future occurrences during the forecast
    /// horizon.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you specify special dates that affect your data differently than regular days.
    ///
    /// The default empty list means:
    /// - No special holidays or events will be explicitly modeled
    /// - The model will rely on its other components to capture any unusual patterns
    ///
    /// Think of holidays like special exceptions to normal patterns:
    /// - Black Friday might cause a huge spike in retail sales
    /// - July 4th might cause a drop in workplace productivity
    /// - Product launch dates might create jumps in your metrics
    ///
    /// Examples of dates you might include:
    /// - Major holidays (Christmas, Thanksgiving, New Year's)
    /// - Business events (Black Friday, Cyber Monday)
    /// - Company-specific dates (product launches, major announcements)
    /// - Unexpected events that affected your metrics (system outages, major news events)
    ///
    /// You would specify holidays when:
    /// - Certain dates consistently show unusual patterns in your data
    /// - You know future special dates that will likely affect your forecasts
    /// - Normal seasonal patterns don't capture these special events adequately
    ///
    /// For example: `Holidays = new List<DateTime> { new DateTime(2023, 12, 25), new DateTime(2023, 11, 24) }` 
    /// would mark Christmas and Black Friday 2023 as special events in your model.
    /// </para>
    /// </remarks>
    public List<DateTime> Holidays { get; set; } = new List<DateTime>();

    /// <summary>
    /// Gets or sets the initial value for changepoint effects.
    /// </summary>
    /// <value>The initial changepoint value, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets the initial value for the magnitude of changepoint effects.
    /// Changepoints represent moments in time when the trend component changes direction
    /// or slope. The initial changepoint value provides a starting point for the optimization
    /// of these effects. Similar to the initial trend value, Prophet typically estimates
    /// appropriate changepoint effects from the data, so the default value of 0.0 is usually
    /// sufficient. Modifying this parameter is rarely necessary unless there's specific
    /// domain knowledge about initial changepoint behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines the starting assumption about how strong trend changes will be in your data.
    ///
    /// The default value of 0.0 means:
    /// - Prophet starts with a neutral assumption about trend changes
    /// - It will learn the appropriate strength of changepoints from your data
    ///
    /// Think of changepoints like turning points in a journey:
    /// - They represent moments when your data's trend changes direction
    /// - This setting controls the initial assumption about how sharp these turns might be
    ///
    /// In practice:
    /// - This parameter rarely needs adjustment
    /// - Prophet will learn the appropriate changepoint effects from your data
    /// - The default value works well for most forecasting scenarios
    ///
    /// Unless you have very specific knowledge about the initial magnitude of trend changes
    /// in your data, it's best to leave this parameter at its default value.
    /// </para>
    /// </remarks>
    public double InitialChangepointValue { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the number of additional regressor variables to include in the model.
    /// </summary>
    /// <value>The number of regressors, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// This parameter specifies the number of external regressor variables that will be included in the model.
    /// Regressors are additional variables that can help explain variations in the time series beyond
    /// the standard trend, seasonality, and holiday components. These might include factors like pricing,
    /// marketing spend, weather conditions, or other relevant variables. Each regressor adds an additional
    /// term to the additive model, allowing the algorithm to learn how the regressor affects the time series.
    /// When using regressors, corresponding values must be provided for both the historical period and the
    /// forecast horizon.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies how many external factors (beyond time patterns) you want to include in your forecast model.
    ///
    /// The default value of 0 means:
    /// - No external variables will be used in the forecast
    /// - Only time-based patterns and holidays will be considered
    ///
    /// Think of regressors like additional explanatory factors:
    /// - If forecasting ice cream sales, temperature might be a regressor
    /// - If forecasting website traffic, marketing spend might be a regressor
    /// - If forecasting energy usage, weather conditions might be regressors
    ///
    /// You would increase this value when:
    /// - You have data on external factors that influence your target variable
    /// - These factors provide additional information beyond time patterns
    /// - You can provide these values for both past and future periods
    ///
    /// For example, setting `RegressorCount = 2` means you plan to include two external variables 
    /// (like price and marketing spend) that help explain variations in your time series.
    ///
    /// Important: When using regressors, you must provide the values for these variables for both 
    /// the historical period and the forecast horizon.
    /// </para>
    /// </remarks>
    public int RegressorCount { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of periods to forecast into the future.
    /// </summary>
    /// <value>The forecast horizon in periods, defaulting to 30.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many time periods into the future the model will forecast.
    /// The units of this horizon match the frequency of the input time series. For daily data,
    /// a value of 30 would forecast 30 days ahead; for hourly data, it would forecast 30 hours ahead.
    /// The appropriate horizon depends on the specific forecasting needs and the characteristics of
    /// the time series. Longer horizons generally result in increased uncertainty and wider prediction
    /// intervals. The forecast quality typically decreases as the horizon extends further into the future,
    /// particularly for volatile time series or series with changing dynamics.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how far into the future your forecast will extend.
    ///
    /// The default value of 30 means:
    /// - For daily data: forecast 30 days ahead
    /// - For hourly data: forecast 30 hours ahead
    /// - For weekly data: forecast 30 weeks ahead
    ///
    /// Think of it like a weather forecast:
    /// - A 7-day forecast is more reliable than a 30-day forecast
    /// - The further you try to predict, the more uncertainty there is
    /// - Different planning horizons require different forecast lengths
    ///
    /// You might want a longer horizon (like 90 or 365):
    /// - For long-term strategic planning
    /// - When your data has strong, stable patterns
    /// - When you need to see longer-term trends
    ///
    /// You might want a shorter horizon (like 7 or 14):
    /// - For tactical, short-term planning
    /// - When your data is volatile or rapidly changing
    /// - When precision is more important than forecasting far ahead
    ///
    /// Remember that forecast accuracy typically decreases as you extend further into the future,
    /// especially for volatile time series.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 30;

    /// <summary>
    /// Gets or sets the flexibility of the trend changepoints.
    /// </summary>
    /// <value>The changepoint prior scale, defaulting to 0.05.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the flexibility of the trend component, specifically how sensitive the trend is
    /// to changes in slope. A higher value allows the trend to change more rapidly, making it more responsive
    /// to fluctuations in the data but potentially more prone to overfitting. A lower value enforces a more
    /// rigid trend with fewer changes, which may generalize better but might miss genuine trend shifts.
    /// This parameter acts as a regularization term - larger values allow more flexibility (less regularization)
    /// while smaller values enforce more rigidity (more regularization). The appropriate value depends on
    /// the characteristics of the time series and the balance needed between fitting historical changes
    /// and producing stable forecasts.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how easily the model's trend can change direction.
    ///
    /// The default value of 0.05 means:
    /// - The trend is moderately flexible
    /// - It can adapt to changes but won't overreact to every fluctuation
    ///
    /// Think of it like steering sensitivity in a car:
    /// - A high value (like 0.5) is like sensitive steering that responds to small movements
    /// - A low value (like 0.01) is like stiff steering that requires more force to turn
    /// - The default (0.05) provides balanced responsiveness
    ///
    /// You might want a higher value (like 0.1 or 0.5):
    /// - When your data shows frequent, genuine changes in trend
    /// - When capturing quick shifts in direction is important
    /// - When you have high confidence in your data quality
    ///
    /// You might want a lower value (like 0.01 or 0.001):
    /// - When you want a smoother, more stable trend line
    /// - When your data contains noise that shouldn't affect the underlying trend
    /// - When you prefer a more conservative forecast
    ///
    /// This is one of the most important tuning parameters in Prophet. Adjusting it can 
    /// significantly change how your forecast responds to historical changes in trend.
    /// </para>
    /// </remarks>
    public double ChangePointPriorScale { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the strength of the seasonality components.
    /// </summary>
    /// <value>The seasonality prior scale, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the flexibility of the seasonality components in the model.
    /// It acts as a regularization term for the seasonality - larger values allow the seasonality
    /// to fit larger fluctuations (less regularization), while smaller values dampen the seasonality
    /// (more regularization). A higher value allows the model to capture stronger seasonal patterns
    /// but increases the risk of fitting to noise. A lower value produces more conservative seasonal
    /// estimates that may miss some genuine seasonal variations but are less likely to overfit.
    /// The default value of 10.0 is generally appropriate for datasets with clear seasonal patterns,
    /// but tuning may be necessary depending on the specific characteristics of the time series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly seasonal patterns can affect your forecast.
    ///
    /// The default value of 10.0 means:
    /// - Seasonal patterns are given significant flexibility
    /// - The model can capture strong seasonal fluctuations
    ///
    /// Think of it like the volume control for seasonality:
    /// - A higher value (like 20.0) turns up the "volume" of seasonal patterns
    /// - A lower value (like 1.0) turns down this "volume"
    /// - The default (10.0) is a balanced setting that works well for most data
    ///
    /// You might want a higher value (like 20.0 or 50.0):
    /// - When your data has very strong, consistent seasonal patterns
    /// - When capturing the full magnitude of seasonal swings is critical
    /// - When seasonal effects clearly dominate other patterns
    ///
    /// You might want a lower value (like 1.0 or 0.1):
    /// - When you want more conservative seasonal estimates
    /// - When the seasonal pattern exists but is subtle
    /// - When you suspect some seasonal-looking patterns might be noise
    ///
    /// Adjusting this parameter changes how much of the variation in your data gets attributed to
    /// seasonal patterns versus other components like trend or noise.
    /// </para>
    /// </remarks>
    public double SeasonalityPriorScale { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the strength of the holiday effects.
    /// </summary>
    /// <value>The holiday prior scale, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the flexibility of the holiday components in the model, similar to
    /// the seasonality prior scale but specifically for holiday effects. It determines how much
    /// of the variation in the time series can be attributed to holidays or special events. A higher
    /// value allows holidays to have stronger effects on the forecast, while a lower value constrains
    /// these effects. This parameter is particularly important when the time series is strongly
    /// affected by holidays or when the holiday effects vary significantly from year to year.
    /// The appropriate value depends on the specific domain and the importance of holiday effects
    /// in the time series being modeled.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly holiday events can affect your forecast.
    ///
    /// The default value of 10.0 means:
    /// - Holiday effects are given significant flexibility
    /// - The model can capture strong deviations on special dates
    ///
    /// Think of it like special exceptions to the rules:
    /// - A higher value (like 20.0) allows holidays to have dramatic effects
    /// - A lower value (like 1.0) constrains holidays to have more modest effects
    /// - The default (10.0) allows for significant but not extreme holiday impacts
    ///
    /// You might want a higher value (like 20.0 or 50.0):
    /// - When holidays cause massive spikes or drops in your data
    /// - When holiday effects vary greatly from year to year
    /// - For retail, e-commerce, or travel data where holidays have outsize importance
    ///
    /// You might want a lower value (like 1.0 or 0.1):
    /// - When holidays have only minor effects on your data
    /// - When you want more conservative holiday adjustments
    /// - When you've specified many holidays and want to prevent overfitting
    ///
    /// This parameter only matters if you've specified holidays using the Holidays parameter.
    /// Otherwise, it has no effect on the model.
    /// </para>
    /// </remarks>
    public double HolidayPriorScale { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets a value indicating whether yearly seasonality should be included in the model.
    /// </summary>
    /// <value>A boolean indicating whether to include yearly seasonality, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines whether the model should include a yearly (annual) seasonality component.
    /// When enabled, Prophet will model patterns that repeat on a yearly cycle (365.25 days). Yearly
    /// seasonality is appropriate for time series that exhibit annual patterns, such as retail sales
    /// with holiday seasons, energy consumption with seasonal weather variations, or tourism with seasonal
    /// travel patterns. Enabling yearly seasonality requires at least one year of historical data for
    /// reliable estimation. If less data is available or if the time series does not exhibit annual
    /// patterns, this component should be disabled to prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether your forecast should include patterns that repeat every year.
    ///
    /// The default value of true means:
    /// - The model will look for and include yearly repeating patterns
    /// - These patterns capture phenomena like holiday seasons, summer increases, winter decreases, etc.
    ///
    /// Think of yearly seasonality like annual climate patterns:
    /// - Summer might always bring higher ice cream sales
    /// - Winter might always show higher heating costs
    /// - December typically shows holiday shopping effects
    ///
    /// You should keep this enabled (true) when:
    /// - Your data spans at least 1-2 years
    /// - You can see clear yearly patterns in your data
    /// - Your business or domain has known annual cycles
    ///
    /// You might want to disable it (set to false) when:
    /// - You have less than a year of historical data
    /// - Your data doesn't show yearly patterns
    /// - You want to simplify your model
    /// - You're dealing with very high-frequency data where yearly patterns are less relevant
    ///
    /// Yearly seasonality is one of the most common patterns in business and economic data.
    /// </para>
    /// </remarks>
    public bool YearlySeasonality { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether weekly seasonality should be included in the model.
    /// </summary>
    /// <value>A boolean indicating whether to include weekly seasonality, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines whether the model should include a weekly seasonality component.
    /// When enabled, Prophet will model patterns that repeat on a weekly cycle (7 days). Weekly 
    /// seasonality is appropriate for time series that exhibit day-of-week effects, such as higher
    /// website traffic on weekdays versus weekends, different retail sales patterns across the week,
    /// or regular weekly business cycles. Enabling weekly seasonality requires at least one week of
    /// historical data for estimation, though several weeks are recommended for reliable results.
    /// For data that does not have a weekly pattern or for data collected less frequently than daily,
    /// this component should be disabled.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether your forecast should include patterns that repeat every week.
    ///
    /// The default value of true means:
    /// - The model will look for and include weekly repeating patterns
    /// - These patterns capture phenomena like weekend drops in office activity, Monday peaks in emails, etc.
    ///
    /// Think of weekly seasonality like the rhythm of a typical week:
    /// - Restaurants might be busier on weekends
    /// - Office buildings use less energy on weekends
    /// - Monday morning might always have higher email volume
    /// - Friday afternoon might show decreased productivity
    ///
    /// You should keep this enabled (true) when:
    /// - Your data is daily or hourly
    /// - You can see different patterns for different days of the week
    /// - Your business or activity has known weekly cycles
    ///
    /// You might want to disable it (set to false) when:
    /// - Your data is collected less frequently than daily
    /// - Your data doesn't show weekly patterns
    /// - You're forecasting something that doesn't vary by day of week
    /// - You want to simplify your model
    ///
    /// Weekly seasonality is very common in business operations, website traffic, and many other daily activities.
    /// </para>
    /// </remarks>
    public bool WeeklySeasonality { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether daily seasonality should be included in the model.
    /// </summary>
    /// <value>A boolean indicating whether to include daily seasonality, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines whether the model should include a daily seasonality component.
    /// When enabled, Prophet will model patterns that repeat on a daily cycle (24 hours). Daily
    /// seasonality is appropriate for time series with sub-daily (e.g., hourly) data that exhibits
    /// time-of-day effects, such as hourly website traffic, energy consumption, or call center volume.
    /// Enabling daily seasonality requires data collected at a sub-daily frequency and generally
    /// several days of historical data for reliable estimation. The default is false because daily
    /// seasonality is only relevant for sub-daily data, which is less common than daily data.
    /// Enabling this for data that is not sub-daily will lead to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether your forecast should include patterns that repeat every day.
    ///
    /// The default value of false means:
    /// - The model will NOT automatically look for daily repeating patterns
    /// - This is appropriate for most data that is collected daily or less frequently
    ///
    /// Think of daily seasonality like the rhythm within a single day:
    /// - Higher coffee shop sales in morning and lunch hours
    /// - Peak electricity usage in evening hours
    /// - Higher website traffic during business hours
    /// - Lower activity during overnight hours
    ///
    /// You should enable this (set to true) when:
    /// - Your data is collected multiple times per day (hourly, every 15 minutes, etc.)
    /// - You can see clear patterns that repeat within each day
    /// - You're forecasting something with known time-of-day effects
    ///
    /// You should keep this disabled (false) when:
    /// - Your data is collected daily or less frequently (weekly, monthly)
    /// - You don't have enough data points within each day to establish patterns
    /// - Your data doesn't have time-of-day effects
    ///
    /// Important: Only enable daily seasonality if you have sub-daily data (multiple observations per day).
    /// Otherwise, it can cause overfitting.
    /// </para>
    /// </remarks>
    public bool DailySeasonality { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether the model should attempt to optimize its parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the model may run an additional optimization pass to refine parameter values beyond
    /// the initial component estimates derived from the data.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Turning this on tells the model to spend extra time trying to find better settings for trend,
    /// seasonality, and other components. This can improve accuracy, but training may take longer.
    /// </para>
    /// </remarks>
    public bool OptimizeParameters { get; set; } = false;

    /// <summary>
    /// Gets or sets the optimizer to use for parameter fitting.
    /// </summary>
    /// <value>The optimizer instance, defaulting to null (uses the default optimizer).</value>
    /// <remarks>
    /// <para>
    /// This parameter allows specification of a custom optimization algorithm for fitting the Prophet model
    /// parameters. When set to null, Prophet uses its default optimization method, typically L-BFGS for
    /// maximum a posteriori estimation. Custom optimizers can be provided to use alternative optimization
    /// approaches, which may be useful for specific types of time series or when special convergence
    /// properties are desired. The optimizer is responsible for finding the optimal values of the model
    /// parameters given the observed data and the prior distributions. Custom optimization can be useful
    /// for very large datasets, complex models, or when specific convergence behavior is required.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you specify a different algorithm for finding the best model parameters.
    ///
    /// The default value of null means:
    /// - Prophet will use its built-in optimizer
    /// - This works well for most forecasting problems
    ///
    /// Think of an optimizer like different strategies for finding the lowest point in a valley:
    /// - The default strategy works well in most terrain
    /// - But sometimes specialized strategies work better for unusual landscapes
    ///
    /// You might want to specify a custom optimizer when:
    /// - You have a very complex forecasting problem
    /// - The default optimizer is not converging well
    /// - You have domain-specific requirements for how parameters are optimized
    /// - You're working with extremely large datasets and need performance optimizations
    ///
    /// For most users, leaving this as null is recommended. Custom optimizers are an advanced feature
    /// that requires understanding of optimization algorithms and their properties.
    /// </para>
    /// </remarks>
    public IOptimizer<T, Matrix<T>, Vector<T>>? Optimizer { get; set; } = null;

    /// <summary>
    /// Gets or sets the number of Fourier terms used for modeling seasonality.
    /// </summary>
    /// <value>The Fourier order, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the complexity of the seasonal components by specifying the number
    /// of Fourier terms used to model each seasonality. Higher orders allow for more complex seasonal
    /// patterns, while lower orders enforce simpler, smoother seasonality. The Fourier order effectively
    /// controls the flexibility of the seasonality - higher orders can capture multiple peaks and troughs
    /// within a single seasonal period, while order 1 can only model a simple sine wave with a single peak
    /// and trough. The appropriate value depends on the complexity of the seasonal patterns in the data.
    /// Overly high values can lead to overfitting, while too low values might miss important seasonal features.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how complex the seasonal patterns can be in your forecast.
    ///
    /// The default value of 3 means:
    /// - Seasonal patterns can have moderate complexity
    /// - The model can capture up to 3 peaks and valleys within each seasonal cycle
    ///
    /// Think of Fourier order like drawing tools for waves:
    /// - Order 1: You can only draw simple waves with one peak and one valley
    /// - Order 3: You can draw more complex waves with multiple bumps and dips
    /// - Order 10: You can capture very intricate patterns with many fluctuations
    ///
    /// You might want a higher order (like 5 or 10):
    /// - When your seasonal patterns are complex with multiple ups and downs
    /// - When you have plenty of historical data to establish these patterns reliably
    /// - When capturing the exact shape of seasonality is important
    ///
    /// You might want a lower order (like 1 or 2):
    /// - When you want simpler, smoother seasonal patterns
    /// - When you have limited historical data
    /// - When you suspect some of the fluctuations in your data are noise rather than true seasonality
    /// - When you want to prevent overfitting
    ///
    /// This setting applies to all seasonality components (yearly, weekly, daily, and custom).
    /// Higher orders require more data to fit reliably.
    /// </para>
    /// </remarks>
    public int FourierOrder { get; set; } = 3;

    /// <summary>
    /// Gets or sets the list of specific changepoints to include in the model.
    /// </summary>
    /// <value>A list of explicit changepoints, defaulting to an empty list.</value>
    /// <remarks>
    /// <para>
    /// This parameter allows explicit specification of points in time where the trend is allowed to change.
    /// By default, Prophet automatically places changepoints uniformly throughout the first 80% of the time series
    /// and then determines their significance. However, when domain knowledge indicates specific events or
    /// interventions that caused changes in the trend, explicitly specifying these changepoints can improve model
    /// accuracy. When provided, these explicit changepoints replace the automatically detected ones. Each changepoint
    /// should be a time point of the same type as the time series data (typically DateTime for time series).
    /// This approach is particularly valuable when specific external events (like policy changes, market disruptions,
    /// or strategic shifts) are known to have affected the time series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you manually specify exactly when important changes occurred in your data's trend.
    ///
    /// The default empty list means:
    /// - Prophet will automatically detect where the trend changes
    /// - It places potential changepoints throughout the first 80% of your historical data
    ///
    /// Think of changepoints like turning points in a journey:
    /// - They mark where your data's path changed direction
    /// - For example, when a business launched a new product, entered a new market, or faced a major disruption
    ///
    /// You would specify explicit changepoints when:
    /// - You know exactly when important events occurred that changed your data's trajectory
    /// - You want to ensure the model captures specific known turning points
    /// - You want to override the automatic changepoint detection
    ///
    /// Examples of changepoints you might specify:
    /// - The date of a major product launch
    /// - When a competitor entered the market
    /// - When a significant policy or pricing change was implemented
    /// - The beginning of a pandemic or economic disruption
    ///
    /// For example: `Changepoints = new List<DateTime> { new DateTime(2020, 3, 15), new DateTime(2021, 6, 1) }` 
    /// would tell the model that important trend changes occurred on March 15, 2020 and June 1, 2021.
    ///
    /// This is an advanced feature that's most useful when you have strong domain knowledge about
    /// significant events affecting your time series.
    /// </para>
    /// </remarks>
    public List<T> Changepoints { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets whether to apply transformations to the predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the model will apply a transformation function to each prediction.
    /// This can be useful for scaling, normalizing, or otherwise adjusting predictions
    /// before returning them to the user.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the model should modify
    /// its predictions before returning them. For example, if your data was transformed
    /// before training (like taking the logarithm to handle skewed data), you might need
    /// to reverse that transformation on the predictions to get meaningful values.
    /// </para>
    /// </remarks>
    public bool ApplyTransformation { get; set; } = false;

    /// <summary>
    /// Gets or sets the function used to transform predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This function is applied to each prediction when ApplyTransformation is enabled.
    /// It can be used to perform operations like scaling, normalizing, or applying
    /// domain-specific transformations to predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This is the specific formula or operation the model uses
    /// to adjust its predictions when ApplyTransformation is turned on. For example, 
    /// if your original data was in dollars but you trained the model on data converted to
    /// thousands of dollars, this function could multiply the predictions by 1000 to
    /// convert them back to dollars.
    /// </para>
    /// </remarks>
    public Func<T, T> TransformPrediction { get; set; } = (x) => x;
}
