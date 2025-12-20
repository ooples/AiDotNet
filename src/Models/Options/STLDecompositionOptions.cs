namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Seasonal-Trend-Loess (STL) decomposition, a versatile method
/// for decomposing time series into seasonal, trend, and residual components.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the decomposition.</typeparam>
/// <remarks>
/// <para>
/// STL (Seasonal-Trend decomposition using Loess) is a robust method for decomposing time series data into 
/// three components: seasonal, trend, and remainder (residual). It uses iterative Loess smoothing (locally 
/// estimated scatterplot smoothing) to extract these components, making it flexible for handling a wide 
/// variety of seasonal patterns and trends. The algorithm is particularly valuable for time series with 
/// complex seasonality, non-linear trends, or outliers. This class provides extensive configuration options 
/// for controlling the STL decomposition process, including parameters for the seasonal and trend components, 
/// robustness iterations, window sizes, and additional adjustments for calendar effects. These options allow 
/// fine-tuning of the decomposition to match the specific characteristics of the time series being analyzed.
/// </para>
/// <para><b>For Beginners:</b> STL decomposition helps break down your time series into meaningful components.
/// 
/// When analyzing time series data:
/// - It's often useful to separate different patterns in the data
/// - STL decomposition splits your data into three parts:
///   1. Seasonal component: Repeating patterns (daily, weekly, monthly, etc.)
///   2. Trend component: Long-term direction (increasing, decreasing, etc.)
///   3. Residual component: What remains after removing season and trend
/// 
/// This separation helps you:
/// - Understand the underlying patterns in your data
/// - Identify anomalies that don't fit the patterns
/// - Make better forecasts by modeling each component separately
/// - Remove seasonality to focus on the trend
/// 
/// STL uses a technique called LOESS (locally estimated scatterplot smoothing) to
/// extract these components in a flexible way that works for many different types of data.
/// 
/// This class lets you configure exactly how the decomposition works to best match your data.
/// </para>
/// </remarks>
public class STLDecompositionOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of time points in one seasonal cycle.
    /// </summary>
    /// <value>A positive integer, defaulting to 12 (monthly seasonality).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of time points that make up one complete seasonal cycle in the data. 
    /// It represents the periodicity of the seasonal pattern. The default value of 12 is appropriate for monthly 
    /// data with yearly seasonality, which is common in many business and economic applications. Other common 
    /// values include 4 for quarterly data with yearly seasonality, 7 for daily data with weekly seasonality, 
    /// or 24 for hourly data with daily seasonality. This property overrides the SeasonalPeriod property 
    /// inherited from the base class to provide a more appropriate default value for STL decomposition. The 
    /// correct specification of the seasonal period is crucial for the algorithm to properly identify and 
    /// extract the seasonal component.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines how long one complete seasonal cycle is in your data.
    /// 
    /// The seasonal period tells the algorithm:
    /// - How many data points make up one complete cycle
    /// - What pattern to look for when extracting seasonality
    /// 
    /// The default value of 12 means:
    /// - The algorithm expects a pattern that repeats every 12 data points
    /// - This is perfect for monthly data with a yearly seasonal pattern
    /// 
    /// Common values include:
    /// - 12: Monthly data with yearly seasonality
    /// - 4: Quarterly data with yearly seasonality
    /// - 7: Daily data with weekly seasonality
    /// - 24: Hourly data with daily seasonality
    /// - 365: Daily data with yearly seasonality
    /// 
    /// When to adjust this value:
    /// - Change it to match the natural cycle in your data
    /// - Must be set correctly for the decomposition to work properly
    /// 
    /// For example, if analyzing hourly website traffic, you might set this to 24 to
    /// capture the daily pattern of activity.
    /// </para>
    /// </remarks>
    public new int SeasonalPeriod { get; set; } = 12;

    /// <summary>
    /// Gets or sets the degree of the polynomial used in the seasonal LOESS smoothing.
    /// </summary>
    /// <value>An integer value of 0 or 1, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the degree of the polynomial used in the LOESS smoothing for the seasonal component. 
    /// A value of 0 uses a constant (flat) local approximation, while a value of 1 (the default) uses a linear 
    /// local approximation. The linear approximation generally provides a better fit to the data, capturing local 
    /// trends within the seasonal component, but may be more sensitive to noise. The constant approximation is 
    /// more robust to noise but may not capture subtle changes in the seasonal pattern. The choice between these 
    /// options depends on the characteristics of the seasonal pattern and the presence of noise in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how flexible the seasonal component can be.
    /// 
    /// When extracting the seasonal component:
    /// - The algorithm fits small local models to segments of data
    /// - This setting determines whether those local models are flat or sloped
    /// 
    /// The default value of 1 means:
    /// - Local linear models are used (can have a slope)
    /// - This allows the seasonal pattern to change more flexibly
    /// 
    /// Your options are:
    /// - 0: Constant/flat local models (more stable, less flexible)
    /// - 1: Linear local models (more flexible, can capture changing patterns)
    /// 
    /// When to adjust this value:
    /// - Set to 0 if your data is noisy and you want a more stable seasonal component
    /// - Keep at 1 (default) if you want to capture subtle changes in the seasonal pattern
    /// 
    /// For example, in retail sales data where seasonal patterns might gradually change
    /// over time, a value of 1 would better capture this evolution.
    /// </para>
    /// </remarks>
    public int SeasonalDegree { get; set; } = 1;

    /// <summary>
    /// Gets or sets the degree of the polynomial used in the trend LOESS smoothing.
    /// </summary>
    /// <value>An integer value of 0 or 1, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the degree of the polynomial used in the LOESS smoothing for the trend component. 
    /// A value of 0 uses a constant (flat) local approximation, while a value of 1 (the default) uses a linear 
    /// local approximation. The linear approximation generally provides a better fit to the data, capturing 
    /// changes in the direction of the trend, but may be more sensitive to noise. The constant approximation is 
    /// more robust to noise but may not capture changes in the trend direction as effectively. The choice between 
    /// these options depends on the characteristics of the trend and the presence of noise in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how flexible the trend component can be.
    /// 
    /// When extracting the trend component:
    /// - The algorithm fits small local models to segments of data
    /// - This setting determines whether those local models are flat or sloped
    /// 
    /// The default value of 1 means:
    /// - Local linear models are used (can have a slope)
    /// - This allows the trend to change direction more flexibly
    /// 
    /// Your options are:
    /// - 0: Constant/flat local models (more stable, less flexible)
    /// - 1: Linear local models (more flexible, can capture changing directions)
    /// 
    /// When to adjust this value:
    /// - Set to 0 if your data is noisy and you want a more stable trend
    /// - Keep at 1 (default) if you want to capture changes in trend direction
    /// 
    /// For example, in economic data where trends might change direction during recessions
    /// and recoveries, a value of 1 would better capture these turning points.
    /// </para>
    /// </remarks>
    public int TrendDegree { get; set; } = 1;

    /// <summary>
    /// Gets or sets the step size for the seasonal LOESS smoothing.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the step size or "jump" used in the LOESS smoothing for the seasonal component. 
    /// It determines how many points to skip when computing the local regressions. A value of 1 (the default) 
    /// means that every point is used, providing the most detailed smoothing. A value greater than 1 means that 
    /// only every n-th point is used, which can significantly reduce computation time for large datasets at the 
    /// cost of some precision. For example, a value of 2 would use every other point, and a value of 3 would use 
    /// every third point. The appropriate value depends on the size of the dataset and the required precision of 
    /// the decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many data points are used when smoothing the seasonal component.
    /// 
    /// During seasonal smoothing:
    /// - The algorithm computes local models at various points
    /// - This setting determines whether it uses every point or skips some
    /// 
    /// The default value of 1 means:
    /// - The algorithm computes a local model at every data point
    /// - This provides the most detailed seasonal component
    /// 
    /// Higher values like 2 or 3 mean:
    /// - The algorithm skips points (every 2nd or 3rd point)
    /// - This makes computation faster but less precise
    /// 
    /// When to adjust this value:
    /// - Increase it (to 2 or higher) for very large datasets to speed up computation
    /// - Keep at 1 (default) for most applications where precision is important
    /// 
    /// For example, with a 10-year daily dataset (3,650 points), setting this to 2 or 3
    /// might significantly speed up computation with minimal loss in quality.
    /// </para>
    /// </remarks>
    public int SeasonalJump { get; set; } = 1;

    /// <summary>
    /// Gets or sets the step size for the trend LOESS smoothing.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the step size or "jump" used in the LOESS smoothing for the trend component. 
    /// It determines how many points to skip when computing the local regressions. A value of 1 (the default) 
    /// means that every point is used, providing the most detailed smoothing. A value greater than 1 means that 
    /// only every n-th point is used, which can significantly reduce computation time for large datasets at the 
    /// cost of some precision. For example, a value of 2 would use every other point, and a value of 3 would use 
    /// every third point. The appropriate value depends on the size of the dataset and the required precision of 
    /// the decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many data points are used when smoothing the trend component.
    /// 
    /// During trend smoothing:
    /// - The algorithm computes local models at various points
    /// - This setting determines whether it uses every point or skips some
    /// 
    /// The default value of 1 means:
    /// - The algorithm computes a local model at every data point
    /// - This provides the most detailed trend component
    /// 
    /// Higher values like 2 or 3 mean:
    /// - The algorithm skips points (every 2nd or 3rd point)
    /// - This makes computation faster but less precise
    /// 
    /// When to adjust this value:
    /// - Increase it (to 2 or higher) for very large datasets to speed up computation
    /// - Keep at 1 (default) for most applications where precision is important
    /// 
    /// For example, with a 10-year daily dataset (3,650 points), setting this to 2 or 3
    /// might significantly speed up computation with minimal loss in quality.
    /// </para>
    /// </remarks>
    public int TrendJump { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of passes through the inner loop of the STL algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of passes through the inner loop of the STL algorithm. The inner loop 
    /// consists of a sequence of smoothing operations that extract the seasonal and trend components. Multiple 
    /// passes through this loop help refine these components, particularly in the presence of complex patterns. 
    /// The default value of 2 provides a good balance between refinement and computational efficiency for many 
    /// applications. A higher value might provide more refined decomposition but requires more computation time. 
    /// A value of 1 might be sufficient for simpler time series or when computational resources are limited.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many times the algorithm refines its estimates of seasonal and trend components.
    /// 
    /// The STL algorithm works iteratively:
    /// - It alternates between estimating seasonal and trend components
    /// - Each pass through the inner loop refines these estimates
    /// 
    /// The default value of 2 means:
    /// - The algorithm makes two passes through the inner loop
    /// - This provides good refinement for most applications
    /// 
    /// Think of it like this:
    /// - More passes: More refined decomposition but more computation time
    /// - Fewer passes: Faster computation but potentially less precise decomposition
    /// 
    /// When to adjust this value:
    /// - Increase it (to 3 or higher) for complex seasonal patterns that need more refinement
    /// - Decrease it to 1 for simpler patterns or when speed is more important than precision
    /// 
    /// For example, with complex economic data that has evolving seasonal patterns,
    /// increasing this to 3 or 4 might provide a more accurate decomposition.
    /// </para>
    /// </remarks>
    public int InnerLoopPasses { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of robust iterations in the STL algorithm.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of robust iterations in the STL algorithm. Robust iterations help 
    /// mitigate the influence of outliers by down-weighting them in subsequent passes through the algorithm. 
    /// A value of 0 disables robust fitting, while a value of 1 or more enables it. The default value of 1 
    /// provides a basic level of robustness suitable for many applications. Higher values provide more robustness 
    /// against outliers but require more computation time. Robust fitting is particularly valuable when the time 
    /// series contains anomalies or outliers that should not influence the seasonal and trend components.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the algorithm handles outliers or unusual data points.
    /// 
    /// Robust iterations:
    /// - Help prevent outliers from distorting the seasonal and trend components
    /// - Work by identifying unusual points and reducing their influence
    /// 
    /// The default value of 1 means:
    /// - The algorithm performs one robust iteration
    /// - This provides basic protection against outliers
    /// 
    /// Think of it like this:
    /// - 0: No robustness (outliers can significantly influence the results)
    /// - 1: Basic robustness (default, good for most applications)
    /// - 2 or more: Increased robustness (better handling of severe outliers)
    /// 
    /// When to adjust this value:
    /// - Increase it when your data contains many outliers or anomalies
    /// - Set to 0 when you're certain your data has no outliers or when you want to preserve all variations
    /// 
    /// For example, in retail sales data with occasional promotional spikes,
    /// increasing this to 2 or 3 would help prevent these spikes from distorting
    /// the underlying seasonal pattern.
    /// </para>
    /// </remarks>
    public int RobustIterations { get; set; } = 1;

    /// <summary>
    /// Gets or sets the threshold for robust weights in outlier detection.
    /// </summary>
    /// <value>A positive double value between 0 and 1, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the threshold for determining which points are considered outliers in the robust 
    /// fitting process. Points with residuals that result in weights below this threshold are treated as severe 
    /// outliers. The default value of 0.001 means that points with weights less than 0.1% of the maximum weight 
    /// are considered severe outliers. A smaller value is more lenient, treating fewer points as severe outliers, 
    /// while a larger value is more strict, treating more points as severe outliers. This parameter is only relevant 
    /// when RobustIterations is greater than 0. The appropriate value depends on the expected frequency and severity 
    /// of outliers in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how extreme a point must be to be considered an outlier.
    /// 
    /// During robust iterations:
    /// - The algorithm calculates weights for each data point
    /// - Points that don't fit the pattern well receive lower weights
    /// - This threshold determines which points are considered severe outliers
    /// 
    /// The default value of 0.001 means:
    /// - Points with weights below 0.1% of the maximum weight are treated as severe outliers
    /// - This provides a reasonable balance for most applications
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.0001): More lenient, fewer points treated as severe outliers
    /// - Higher values (e.g., 0.01): More strict, more points treated as severe outliers
    /// 
    /// When to adjust this value:
    /// - Decrease it when you want to focus only on the most extreme outliers
    /// - Increase it when you want to be more aggressive in identifying outliers
    /// 
    /// This setting only matters when RobustIterations is greater than 0.
    /// </para>
    /// </remarks>
    public double RobustWeightThreshold { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the bandwidth parameter for the seasonal LOESS smoothing.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.75.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the bandwidth parameter for the LOESS smoothing of the seasonal component. The 
    /// bandwidth determines the proportion of data points used in each local regression, controlling the smoothness 
    /// of the fitted curve. A value closer to 1 uses more points in each local regression, resulting in a smoother 
    /// seasonal component. A value closer to 0 uses fewer points, allowing for more local variation but potentially 
    /// capturing noise. The default value of 0.75 provides a moderate level of smoothing suitable for many applications. 
    /// The appropriate value depends on the characteristics of the seasonal pattern and the presence of noise in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how smooth the seasonal component will be.
    /// 
    /// The seasonal bandwidth determines:
    /// - What proportion of nearby points are used in each local regression
    /// - How smooth vs. flexible the seasonal component will be
    /// 
    /// The default value of 0.75 means:
    /// - Each local regression uses 75% of the data points
    /// - This creates a moderately smooth seasonal component
    /// 
    /// Think of it like this:
    /// - Higher values (closer to 1): Smoother seasonal component, less responsive to local variations
    /// - Lower values (closer to 0): More flexible seasonal component, more responsive to local variations
    /// 
    /// When to adjust this value:
    /// - Increase it when you want a smoother, more stable seasonal pattern
    /// - Decrease it when you want to capture more detailed seasonal variations
    /// 
    /// For example, in tourism data where seasonal patterns might have sharp peaks,
    /// decreasing this to 0.5 might better capture these distinctive features.
    /// </para>
    /// </remarks>
    public double SeasonalBandwidth { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the bandwidth parameter for the trend LOESS smoothing.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.75.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the bandwidth parameter for the LOESS smoothing of the trend component. The 
    /// bandwidth determines the proportion of data points used in each local regression, controlling the smoothness 
    /// of the fitted curve. A value closer to 1 uses more points in each local regression, resulting in a smoother 
    /// trend component. A value closer to 0 uses fewer points, allowing for more local variation but potentially 
    /// capturing noise. The default value of 0.75 provides a moderate level of smoothing suitable for many applications. 
    /// The appropriate value depends on the characteristics of the trend and the presence of noise in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how smooth the trend component will be.
    /// 
    /// The trend bandwidth determines:
    /// - What proportion of nearby points are used in each local regression
    /// - How smooth vs. flexible the trend component will be
    /// 
    /// The default value of 0.75 means:
    /// - Each local regression uses 75% of the data points
    /// - This creates a moderately smooth trend component
    /// 
    /// Think of it like this:
    /// - Higher values (closer to 1): Smoother trend, less responsive to local variations
    /// - Lower values (closer to 0): More flexible trend, more responsive to local variations
    /// 
    /// When to adjust this value:
    /// - Increase it when you want a smoother, more stable trend
    /// - Decrease it when you want to capture more detailed trend variations
    /// 
    /// For example, in economic data where you want to capture the overall direction
    /// while ignoring short-term fluctuations, increasing this to 0.9 might be appropriate.
    /// </para>
    /// </remarks>
    public double TrendBandwidth { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the bandwidth parameter for the low-pass filter LOESS smoothing.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.75.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the bandwidth parameter for the LOESS smoothing used in the low-pass filter step of 
    /// the STL algorithm. The low-pass filter helps separate the seasonal and trend components by smoothing the 
    /// detrended series. The bandwidth determines the proportion of data points used in each local regression, 
    /// controlling the smoothness of the fitted curve. A value closer to 1 uses more points in each local regression, 
    /// resulting in a smoother filter. A value closer to 0 uses fewer points, allowing for more local variation. 
    /// The default value of 0.75 provides a moderate level of smoothing suitable for many applications. The appropriate 
    /// value depends on the characteristics of the time series and the desired separation between seasonal and trend 
    /// components.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the algorithm separates seasonal and trend components.
    /// 
    /// The low-pass bandwidth determines:
    /// - How the algorithm filters out high-frequency variations
    /// - How clearly seasonal and trend components are separated
    /// 
    /// The default value of 0.75 means:
    /// - Each local regression in the low-pass filter uses 75% of the data points
    /// - This creates a moderate separation between seasonal and trend components
    /// 
    /// Think of it like this:
    /// - Higher values (closer to 1): Stronger separation between seasonal and trend components
    /// - Lower values (closer to 0): Less distinct separation between components
    /// 
    /// When to adjust this value:
    /// - Increase it when you want clearer separation between seasonal and trend components
    /// - Decrease it when the default separation seems too aggressive
    /// 
    /// This is a more technical parameter that most users won't need to adjust unless
    /// they're experiencing specific issues with component separation.
    /// </para>
    /// </remarks>
    public double LowPassBandwidth { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the window size for the trend component smoothing.
    /// </summary>
    /// <value>A positive integer, defaulting to 18 (1.5 times the default seasonal period).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the window size for smoothing the trend component. It determines how many consecutive 
    /// data points are considered in each local regression for the trend. A larger window results in a smoother trend 
    /// component, while a smaller window allows the trend to follow more local variations. The default value of 18 
    /// (which is 1.5 times the default seasonal period of 12) provides a good balance for many applications with 
    /// monthly data. For time series with different seasonal periods or different requirements for trend smoothness, 
    /// this value might need adjustment. A common rule of thumb is to set this to 1.5 times the seasonal period, 
    /// which is reflected in the default value.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many data points are used when smoothing the trend.
    /// 
    /// The trend window size determines:
    /// - How many consecutive points are used in each local regression for the trend
    /// - How smooth vs. responsive the trend component will be
    /// 
    /// The default value of 18 means:
    /// - Each local regression for the trend uses 18 consecutive data points
    /// - For monthly data, this spans 1.5 years, which is a good balance
    /// 
    /// Think of it like this:
    /// - Larger values: Smoother trend that ignores short-term fluctuations
    /// - Smaller values: More responsive trend that follows shorter-term changes
    /// 
    /// When to adjust this value:
    /// - Increase it when you want a smoother long-term trend
    /// - Decrease it when you want the trend to capture more medium-term changes
    /// - A common rule of thumb is to set it to 1.5 times your seasonal period
    /// 
    /// For example, with quarterly data (seasonal period of 4), you might set this to 6,
    /// while with daily data and weekly seasonality (period of 7), you might set it to 10-11.
    /// </para>
    /// </remarks>
    public int TrendWindowSize { get; set; } = 18;

    /// <summary>
    /// Gets or sets the window size for the seasonal component LOESS smoothing.
    /// </summary>
    /// <value>A positive odd integer, defaulting to 121 (10 times the default seasonal period plus 1).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the window size for the LOESS smoothing of the seasonal component. It determines how 
    /// many data points are considered in each local regression for extracting the seasonal component. The window 
    /// size should be odd to ensure symmetry around the central point. A larger window results in a smoother seasonal 
    /// component, while a smaller window allows for more variation in the seasonal pattern. The default value of 121 
    /// (which is 10 times the default seasonal period of 12, plus 1) provides a good balance for many applications 
    /// with monthly data. For time series with different seasonal periods or different requirements for seasonal 
    /// smoothness, this value might need adjustment. A common rule of thumb is to set this to 10 times the seasonal 
    /// period plus 1, which is reflected in the default value.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the seasonal pattern is extracted from the data.
    /// 
    /// The seasonal LOESS window determines:
    /// - How many data points are used when extracting the seasonal component
    /// - How stable vs. variable the seasonal pattern will be across cycles
    /// 
    /// The default value of 121 means:
    /// - For monthly data, the algorithm looks at about 10 years of data for each seasonal estimate
    /// - This creates a stable seasonal pattern that doesn't change much from year to year
    /// 
    /// Think of it like this:
    /// - Larger values: More stable seasonal pattern that changes very little over time
    /// - Smaller values: More adaptive seasonal pattern that can evolve over time
    /// 
    /// When to adjust this value:
    /// - Increase it when you want a very stable seasonal pattern
    /// - Decrease it when you want to allow the seasonal pattern to evolve over time
    /// - A common rule of thumb is to set it to 10 times your seasonal period plus 1
    /// 
    /// For example, if you believe your seasonal pattern is gradually changing over time,
    /// you might reduce this to 7 times the seasonal period plus 1.
    /// </para>
    /// </remarks>
    public int SeasonalLoessWindow { get; set; } = 121;

    /// <summary>
    /// Gets or sets the window size for the low-pass filter.
    /// </summary>
    /// <value>A positive integer, defaulting to 12 (same as the default seasonal period).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the window size for the low-pass filter used in the STL algorithm. The low-pass filter 
    /// helps separate the seasonal and trend components by smoothing the detrended series. A larger window results in 
    /// a smoother filter, while a smaller window allows for more local variation. The default value of 12 (which is 
    /// the same as the default seasonal period) provides a good balance for many applications with monthly data. For 
    /// time series with different seasonal periods or different requirements for component separation, this value might 
    /// need adjustment. A common practice is to set this equal to the seasonal period, which is reflected in the default 
    /// value.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the algorithm filters out high-frequency variations.
    /// 
    /// The low-pass filter window determines:
    /// - How many consecutive points are used in the smoothing filter
    /// - How effectively high-frequency variations are removed
    /// 
    /// The default value of 12 means:
    /// - For monthly data, the filter uses a 12-month window
    /// - This aligns with the seasonal period, which works well for most applications
    /// 
    /// Think of it like this:
    /// - Larger values: Stronger filtering of high-frequency variations
    /// - Smaller values: Less filtering, allowing more high-frequency components
    /// 
    /// When to adjust this value:
    /// - Usually best to keep this equal to your seasonal period
    /// - Adjust only if you have specific requirements for component separation
    /// 
    /// This is a technical parameter that most users won't need to adjust unless
    /// they're experiencing specific issues with the decomposition.
    /// </para>
    /// </remarks>
    public int LowPassFilterWindowSize { get; set; } = 12;

    /// <summary>
    /// Gets or sets the window size for the trend LOESS smoothing.
    /// </summary>
    /// <value>A positive integer, defaulting to 12 (same as the default seasonal period).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the window size for the LOESS smoothing of the trend component. It determines how 
    /// many consecutive data points are considered in each local regression for the trend. This is an alternative 
    /// to using the TrendWindowSize property and might be used in certain implementations of the STL algorithm. 
    /// A larger window results in a smoother trend component, while a smaller window allows the trend to follow 
    /// more local variations. The default value of 12 (which is the same as the default seasonal period) provides 
    /// a moderate level of smoothing suitable for many applications with monthly data. For time series with different 
    /// seasonal periods or different requirements for trend smoothness, this value might need adjustment.
    /// </para>
    /// <para><b>For Beginners:</b> This setting is an alternative way to control how smooth the trend component will be.
    /// 
    /// The trend LOESS window determines:
    /// - How many consecutive points are used in each local regression for the trend
    /// - How smooth vs. responsive the trend component will be
    /// 
    /// The default value of 12 means:
    /// - Each local regression for the trend uses 12 consecutive data points
    /// - For monthly data, this spans 1 year, which provides moderate smoothing
    /// 
    /// Think of it like this:
    /// - Larger values: Smoother trend that ignores short-term fluctuations
    /// - Smaller values: More responsive trend that follows shorter-term changes
    /// 
    /// When to adjust this value:
    /// - Increase it when you want a smoother long-term trend
    /// - Decrease it when you want the trend to capture more medium-term changes
    /// - This setting may be used instead of TrendWindowSize in certain implementations
    /// 
    /// For example, with quarterly data, you might set this to 4 for a one-year window,
    /// or 8 for a two-year window, depending on how smooth you want the trend to be.
    /// </para>
    /// </remarks>
    public int TrendLoessWindow { get; set; } = 12; // Same as default SeasonalPeriod

    /// <summary>
    /// Gets or sets the type of STL algorithm to use.
    /// </summary>
    /// <value>A value from the STLAlgorithmType enumeration, defaulting to STLAlgorithmType.Standard.</value>
    /// <remarks>
    /// <para>
    /// This property specifies which variant of the STL algorithm to use for the decomposition. Different variants 
    /// might have different approaches to handling the seasonal and trend components, with trade-offs in terms of 
    /// computational efficiency, flexibility, and robustness. The Standard algorithm (the default) is the classical 
    /// STL approach as described in the original literature. Other options might include variants optimized for 
    /// specific types of data or computational constraints. The choice of algorithm can affect both the quality of 
    /// the decomposition and the computational resources required.
    /// </para>
    /// <para><b>For Beginners:</b> This setting selects which version of the STL algorithm to use.
    /// 
    /// Different algorithm types:
    /// - Represent different implementations or variants of the STL method
    /// - May have different trade-offs in terms of speed, accuracy, and flexibility
    /// 
    /// The default Standard algorithm:
    /// - Implements the classical STL approach as described in the original research
    /// - Works well for most applications
    /// 
    /// Other possible options might include:
    /// - Faster variants that sacrifice some accuracy for speed
    /// - More robust variants that handle unusual data better
    /// - Specialized variants for specific types of time series
    /// 
    /// When to adjust this value:
    /// - Keep the default for most applications
    /// - Consider alternatives only if you have specific requirements or constraints
    /// 
    /// This is an advanced setting that most users won't need to change unless they
    /// have specific knowledge about the different algorithm variants.
    /// </para>
    /// </remarks>
    public STLAlgorithmType AlgorithmType { get; set; } = STLAlgorithmType.Standard;

    /// <summary>
    /// Gets or sets the array of dates corresponding to the time series observations.
    /// </summary>
    /// <value>An array of DateTime values, defaulting to an empty array.</value>
    /// <remarks>
    /// <para>
    /// This property provides the dates or timestamps for each observation in the time series. When specified, these 
    /// dates can be used to handle irregular time series (where observations are not equally spaced in time) and to 
    /// incorporate calendar effects into the decomposition. The dates are also useful for visualization and for 
    /// aligning the decomposed components with the original time series. If this property is not set, the time series 
    /// is assumed to be regular with consecutive integer indices. For regular time series with a known start date and 
    /// interval, the StartDate and Interval properties can be used instead of providing the full array of dates.
    /// </para>
    /// <para><b>For Beginners:</b> This setting provides the actual calendar dates for your time series data points.
    /// 
    /// The dates array:
    /// - Associates each data point with its specific date or timestamp
    /// - Helps the algorithm handle irregular time series (uneven spacing between observations)
    /// - Enables calendar-based adjustments (day of week, holidays, etc.)
    /// 
    /// The default empty array means:
    /// - No specific dates are associated with the data points
    /// - The algorithm treats the series as regular with integer indices (0, 1, 2, ...)
    /// 
    /// When to set this value:
    /// - Always provide dates when working with real-world time series data
    /// - Especially important for irregular time series or when using calendar adjustments
    /// - For regular time series, you can alternatively use StartDate and Interval
    /// 
    /// For example, if analyzing monthly sales data, you would provide an array of dates
    /// like [2020-01-01, 2020-02-01, 2020-03-01, ...] to properly align the data with the calendar.
    /// </para>
    /// </remarks>
    public DateTime[] Dates { get; set; } = Array.Empty<DateTime>();

    /// <summary>
    /// Gets or sets the start date of the time series.
    /// </summary>
    /// <value>A nullable DateTime value, defaulting to null.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the date or timestamp of the first observation in the time series. When used together 
    /// with the Interval property, it provides an alternative to the Dates array for regular time series (where 
    /// observations are equally spaced in time). The start date is used to generate the dates for all observations 
    /// by adding multiples of the interval. This approach is more memory-efficient than storing all dates explicitly 
    /// when the time series is regular. If both Dates and StartDate are provided, the Dates array takes precedence.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies when your time series begins.
    /// 
    /// The start date:
    /// - Defines the date of the first observation in your time series
    /// - Works together with the Interval property for regular time series
    /// - Provides a more efficient alternative to listing all dates
    /// 
    /// The default null value means:
    /// - No specific start date is defined
    /// - The algorithm treats the series as having integer indices
    /// 
    /// When to set this value:
    /// - For regular time series (equal spacing between observations)
    /// - When you don't want to provide the full array of dates
    /// - Must be used together with the Interval property
    /// 
    /// For example, if you have monthly data starting from January 2020, you would set
    /// StartDate to 2020-01-01 and Interval to TimeSpan.FromDays(30) or a similar value.
    /// </para>
    /// </remarks>
    public DateTime? StartDate { get; set; }

    /// <summary>
    /// Gets or sets the time interval between consecutive observations in the time series.
    /// </summary>
    /// <value>A nullable TimeSpan value, defaulting to null.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the time interval between consecutive observations in the time series. When used 
    /// together with the StartDate property, it provides an alternative to the Dates array for regular time series 
    /// (where observations are equally spaced in time). The interval is used to generate the dates for all observations 
    /// by adding multiples of the interval to the start date. This approach is more memory-efficient than storing all 
    /// dates explicitly when the time series is regular. Common values include TimeSpan.FromDays(1) for daily data, 
    /// TimeSpan.FromDays(7) for weekly data, or TimeSpan.FromDays(30) for approximately monthly data. If both Dates 
    /// and Interval are provided, the Dates array takes precedence.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies how much time passes between consecutive data points.
    /// 
    /// The interval:
    /// - Defines the time spacing between observations in your time series
    /// - Works together with the StartDate property for regular time series
    /// - Provides a more efficient alternative to listing all dates
    /// 
    /// The default null value means:
    /// - No specific time interval is defined
    /// - The algorithm treats the series as having integer indices
    /// 
    /// When to set this value:
    /// - For regular time series (equal spacing between observations)
    /// - When you don't want to provide the full array of dates
    /// - Must be used together with the StartDate property
    /// 
    /// Common interval values:
    /// - TimeSpan.FromDays(1): Daily data
    /// - TimeSpan.FromDays(7): Weekly data
    /// - TimeSpan.FromDays(30): Monthly data (approximate)
    /// - TimeSpan.FromHours(1): Hourly data
    /// 
    /// For example, if you have daily stock prices, you would set Interval to TimeSpan.FromDays(1).
    /// </para>
    /// </remarks>
    public TimeSpan? Interval { get; set; }

    /// <summary>
    /// Gets or sets whether to adjust for day-of-week effects in the decomposition.
    /// </summary>
    /// <value>A boolean value, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether to incorporate day-of-week effects into the decomposition. When set to true, 
    /// the algorithm will attempt to identify and account for systematic variations associated with different days 
    /// of the week. This is particularly relevant for daily data where certain days (e.g., weekends or Mondays) might 
    /// have consistently different patterns. The day-of-week factors can be pre-specified using the DayOfWeekFactors 
    /// property or estimated from the data. The default value of false disables this adjustment, which is appropriate 
    /// for data that doesn't exhibit day-of-week patterns or for non-daily data. Enabling this adjustment requires 
    /// that dates are provided for the time series, either through the Dates array or through the StartDate and 
    /// Interval properties.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm accounts for different patterns on different days of the week.
    /// 
    /// Day-of-week adjustment:
    /// - Helps identify and separate patterns specific to certain days
    /// - For example, retail sales might always be higher on weekends
    /// - Or website traffic might always drop on Sundays
    /// 
    /// The default false value means:
    /// - No adjustment is made for day-of-week effects
    /// - All days are treated the same
    /// 
    /// When to enable this feature:
    /// - For daily data where different days of the week show consistent patterns
    /// - When you want to separate these weekly patterns from other seasonal patterns
    /// - Requires that dates are provided for your time series
    /// 
    /// For example, if analyzing daily hospital admissions where weekends consistently
    /// have different patterns than weekdays, you would set this to true.
    /// </para>
    /// </remarks>
    public bool AdjustForDayOfWeek { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to adjust for month-of-year effects in the decomposition.
    /// </summary>
    /// <value>A boolean value, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether to incorporate month-of-year effects into the decomposition. When set to true, 
    /// the algorithm will attempt to identify and account for systematic variations associated with different months 
    /// of the year. This is particularly relevant for data where certain months might have consistently different 
    /// patterns beyond the regular seasonal cycle. The month-of-year factors can be pre-specified using the 
    /// MonthOfYearFactors property or estimated from the data. The default value of false disables this adjustment, 
    /// which is appropriate when the regular seasonal component adequately captures monthly variations. Enabling this 
    /// adjustment requires that dates are provided for the time series, either through the Dates array or through the 
    /// StartDate and Interval properties.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm accounts for specific effects of different months.
    /// 
    /// Month-of-year adjustment:
    /// - Helps identify and separate patterns specific to certain months
    /// - Goes beyond regular seasonality to capture month-specific effects
    /// - For example, retail sales might have special patterns in December beyond the winter season effect
    /// 
    /// The default false value means:
    /// - No special adjustment is made for month-specific effects
    /// - Monthly patterns are handled by the regular seasonal component
    /// 
    /// When to enable this feature:
    /// - When certain months consistently show unique patterns beyond regular seasonality
    /// - When you want to separate these specific monthly effects
    /// - Requires that dates are provided for your time series
    /// 
    /// For example, if analyzing sales data where December has unique patterns beyond
    /// the regular winter seasonal effect, you would set this to true.
    /// </para>
    /// </remarks>
    public bool AdjustForMonthOfYear { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to adjust for holiday effects in the decomposition.
    /// </summary>
    /// <value>A boolean value, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether to incorporate holiday effects into the decomposition. When set to true, 
    /// the algorithm will attempt to identify and account for systematic variations associated with holidays or 
    /// special events. This is particularly relevant for data where holidays might cause significant deviations 
    /// from the regular patterns. The holidays and their effects can be pre-specified using the Holidays property 
    /// or estimated from the data. The default value of false disables this adjustment, which is appropriate when 
    /// holidays don't have significant effects or when they're not of interest for the analysis. Enabling this 
    /// adjustment requires that dates are provided for the time series, either through the Dates array or through 
    /// the StartDate and Interval properties.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm accounts for the effects of holidays and special events.
    /// 
    /// Holiday adjustment:
    /// - Helps identify and separate patterns specific to holidays
    /// - For example, retail sales might spike before Christmas
    /// - Or travel data might show unique patterns around major holidays
    /// 
    /// The default false value means:
    /// - No adjustment is made for holiday effects
    /// - Holidays are treated like any other day
    /// 
    /// When to enable this feature:
    /// - When holidays consistently cause significant deviations in your data
    /// - When you want to separate these holiday effects from regular patterns
    /// - Requires that dates are provided for your time series
    /// - Works best when you also specify the Holidays dictionary
    /// 
    /// For example, if analyzing retail sales data where Christmas, Black Friday, and
    /// other shopping holidays create major spikes, you would set this to true.
    /// </para>
    /// </remarks>
    public bool AdjustForHolidays { get; set; } = false;

    /// <summary>
    /// Gets or sets the factors for day-of-week effects.
    /// </summary>
    /// <value>A vector of length 7, defaulting to a new vector of length 7.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factors or coefficients for day-of-week effects when AdjustForDayOfWeek is set to 
    /// true. Each element in the vector corresponds to a day of the week, typically starting with Sunday (index 0) 
    /// through Saturday (index 6), though the exact mapping might depend on the implementation. These factors represent 
    /// the multiplicative or additive effect of each day of the week on the time series values. If not explicitly set, 
    /// these factors might be estimated from the data when AdjustForDayOfWeek is true. The default is an empty vector 
    /// of length 7, indicating that no pre-specified factors are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This setting allows you to specify how different days of the week affect your data.
    /// 
    /// The day-of-week factors:
    /// - Represent the effect of each day of the week on your data
    /// - Are stored as a vector with 7 values (one for each day)
    /// - Can be used to adjust for weekly patterns
    /// 
    /// The default empty vector means:
    /// - No pre-specified day-of-week effects are provided
    /// - If AdjustForDayOfWeek is true, these will be estimated from the data
    /// 
    /// When to set this value:
    /// - When you have prior knowledge about day-of-week effects
    /// - When you want to specify these effects manually rather than having them estimated
    /// - Must be used with AdjustForDayOfWeek set to true
    /// 
    /// For example, if you know that Saturdays typically have values 20% higher than
    /// the average day, you might set the Saturday factor to 1.2.
    /// </para>
    /// </remarks>
    public Vector<T> DayOfWeekFactors { get; set; } = new Vector<T>(7);

    /// <summary>
    /// Gets or sets the factors for month-of-year effects.
    /// </summary>
    /// <value>A vector of length 12, defaulting to a new vector of length 12.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factors or coefficients for month-of-year effects when AdjustForMonthOfYear is set 
    /// to true. Each element in the vector corresponds to a month of the year, typically starting with January (index 0) 
    /// through December (index 11), though the exact mapping might depend on the implementation. These factors represent 
    /// the multiplicative or additive effect of each month on the time series values. If not explicitly set, these 
    /// factors might be estimated from the data when AdjustForMonthOfYear is true. The default is an empty vector of 
    /// length 12, indicating that no pre-specified factors are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This setting allows you to specify how different months of the year affect your data.
    /// 
    /// The month-of-year factors:
    /// - Represent the effect of each month on your data
    /// - Are stored as a vector with 12 values (one for each month)
    /// - Can be used to adjust for monthly patterns beyond regular seasonality
    /// 
    /// The default empty vector means:
    /// - No pre-specified month-of-year effects are provided
    /// - If AdjustForMonthOfYear is true, these will be estimated from the data
    /// 
    /// When to set this value:
    /// - When you have prior knowledge about month-specific effects
    /// - When you want to specify these effects manually rather than having them estimated
    /// - Must be used with AdjustForMonthOfYear set to true
    /// 
    /// For example, if you know that December typically has values 50% higher than
    /// the average month due to holiday effects, you might set the December factor to 1.5.
    /// </para>
    /// </remarks>
    public Vector<T> MonthOfYearFactors { get; set; } = new Vector<T>(12);

    /// <summary>
    /// Gets or sets the dictionary of holiday dates and their effects.
    /// </summary>
    /// <value>A dictionary mapping DateTime values to their effects, defaulting to an empty dictionary.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the holidays or special events and their effects on the time series when AdjustForHolidays 
    /// is set to true. The dictionary keys are the dates of the holidays, and the values represent the multiplicative or 
    /// additive effect of each holiday on the time series values. If not explicitly set, these effects might be estimated 
    /// from the data when AdjustForHolidays is true, though this would typically require the algorithm to know which 
    /// dates are holidays. The default is an empty dictionary, indicating that no pre-specified holidays are provided. 
    /// This property is particularly useful for accounting for the effects of irregular events that don't follow a 
    /// regular calendar pattern, such as Easter, Thanksgiving, or other moving holidays.
    /// </para>
    /// <para><b>For Beginners:</b> This setting allows you to specify how holidays and special events affect your data.
    /// 
    /// The holidays dictionary:
    /// - Maps specific dates to their effects on your data
    /// - Allows you to account for irregular events like holidays
    /// - Can handle both fixed holidays (like Christmas) and moving holidays (like Easter)
    /// 
    /// The default empty dictionary means:
    /// - No pre-specified holiday effects are provided
    /// - If AdjustForHolidays is true, the algorithm will need to identify holidays
    /// 
    /// When to set this value:
    /// - When you have prior knowledge about holiday effects
    /// - When you want to specify these effects manually
    /// - Must be used with AdjustForHolidays set to true
    /// 
    /// For example, you might create entries for:
    /// - Christmas (December 25): 2.0 (double the normal value)
    /// - Black Friday (day after Thanksgiving): 3.0 (triple the normal value)
    /// - July 4th: 0.5 (half the normal value)
    /// 
    /// This is particularly useful for retail, travel, or other data heavily influenced by holidays.
    /// </para>
    /// </remarks>
    public Dictionary<DateTime, T> Holidays { get; set; } = new Dictionary<DateTime, T>();

    /// <summary>
    /// Gets or sets the method used for detecting outliers in the time series.
    /// </summary>
    /// <value>A value from the OutlierDetectionMethod enumeration, defaulting to OutlierDetectionMethod.ZScore.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the statistical method used to identify outliers in the time series. Outliers are 
    /// observations that deviate significantly from the expected pattern and might distort the decomposition if not 
    /// properly handled. Different methods have different approaches to defining what constitutes an outlier. The 
    /// Z-Score method (the default) identifies outliers based on how many standard deviations they are from the mean. 
    /// The IQR (Interquartile Range) method identifies outliers based on how far they are from the first and third 
    /// quartiles. Other methods might include modified Z-score, Tukey's method, or domain-specific approaches. The 
    /// choice of method can affect which observations are identified as outliers and how they are treated in the 
    /// decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how the algorithm identifies unusual data points.
    /// 
    /// The outlier detection method:
    /// - Defines the statistical approach used to identify outliers
    /// - Helps the algorithm handle unusual data points appropriately
    /// 
    /// The default Z-Score method:
    /// - Identifies outliers based on how many standard deviations they are from the mean
    /// - Points beyond the ZScoreThreshold (default 3.0) are considered outliers
    /// 
    /// Common alternatives include:
    /// - IQR (Interquartile Range): Identifies outliers based on the spread of the middle 50% of data
    /// - Modified Z-Score: A more robust version of Z-Score for non-normal distributions
    /// 
    /// When to adjust this value:
    /// - Change to IQR when your data is skewed or has a non-normal distribution
    /// - Keep at Z-Score (default) for most applications with roughly normal data
    /// 
    /// For example, in financial data with extreme values, the IQR method might be
    /// more appropriate as it's less influenced by the extreme values themselves.
    /// </para>
    /// </remarks>
    public OutlierDetectionMethod OutlierDetectionMethod { get; set; } = OutlierDetectionMethod.ZScore;

    /// <summary>
    /// Gets or sets the threshold for identifying outliers using the Z-Score method.
    /// </summary>
    /// <value>A positive double value, defaulting to 3.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the threshold for identifying outliers when using the Z-Score method 
    /// (OutlierDetectionMethod.ZScore). Observations with Z-scores (number of standard deviations from the mean) 
    /// exceeding this threshold in absolute value are considered outliers. The default value of 3.0 is a common 
    /// choice in statistical practice, corresponding to approximately 99.7% of the data in a normal distribution. 
    /// A smaller value would identify more observations as outliers, while a larger value would be more conservative, 
    /// identifying only the most extreme observations as outliers. This property is only relevant when 
    /// OutlierDetectionMethod is set to ZScore.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how extreme a data point must be to be considered an outlier when using the Z-Score method.
    /// 
    /// The Z-Score threshold:
    /// - Defines how many standard deviations from the mean a point must be to be an outlier
    /// - Higher values mean fewer points will be identified as outliers
    /// 
    /// The default value of 3.0 means:
    /// - Points more than 3 standard deviations from the mean are considered outliers
    /// - For normally distributed data, this identifies about 0.3% of points as outliers
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 2.0): More aggressive outlier detection, more points flagged
    /// - Higher values (e.g., 4.0): More conservative detection, only extreme points flagged
    /// 
    /// When to adjust this value:
    /// - Decrease it when you want to be more aggressive in identifying outliers
    /// - Increase it when you want to be more conservative
    /// - Only relevant when OutlierDetectionMethod is set to ZScore
    /// 
    /// For example, in quality control applications where outliers represent defects,
    /// you might use a lower threshold like 2.5 to be more sensitive to potential issues.
    /// </para>
    /// </remarks>
    public double ZScoreThreshold { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the multiplier for the interquartile range (IQR) when identifying outliers using the IQR method.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.5.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the multiplier for the interquartile range (IQR) when using the IQR method 
    /// (OutlierDetectionMethod.IQR) to identify outliers. The IQR is the difference between the third quartile (Q3) 
    /// and the first quartile (Q1) of the data. Observations below Q1 - (IQRMultiplier * IQR) or above 
    /// Q3 + (IQRMultiplier * IQR) are considered outliers. The default value of 1.5 is a common choice in statistical 
    /// practice, corresponding to the "inner fences" in a box plot. A smaller value would identify more observations 
    /// as outliers, while a larger value would be more conservative, identifying only the most extreme observations 
    /// as outliers. This property is only relevant when OutlierDetectionMethod is set to IQR.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how extreme a data point must be to be considered an outlier when using the IQR method.
    /// 
    /// The IQR multiplier:
    /// - Defines how far beyond the quartiles a point must be to be an outlier
    /// - Higher values mean fewer points will be identified as outliers
    /// 
    /// The default value of 1.5 means:
    /// - Points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are considered outliers
    /// - This is the standard definition used in box plots
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 1.0): More aggressive outlier detection, more points flagged
    /// - Higher values (e.g., 2.0): More conservative detection, only extreme points flagged
    /// 
    /// When to adjust this value:
    /// - Decrease it when you want to be more aggressive in identifying outliers
    /// - Increase it when you want to be more conservative
    /// - Only relevant when OutlierDetectionMethod is set to IQR
    /// 
    /// For example, in exploratory data analysis where you want to identify potential
    /// outliers for further investigation, you might use the standard value of 1.5.
    /// </para>
    /// </remarks>
    public double IQRMultiplier { get; set; } = 1.5;
}
