namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for SEATS (Seasonal Extraction in ARIMA Time Series) decomposition.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SEATS (Seasonal Extraction in ARIMA Time Series) is a method used to break down time series data 
/// into different components, making it easier to understand patterns and make predictions.
/// 
/// Think of time series data as a recording of values over time, like daily temperature readings, monthly sales figures, 
/// or quarterly economic indicators. This data often contains several patterns mixed together:
/// 
/// 1. Trend: The long-term direction (going up, down, or staying flat)
/// 2. Seasonal patterns: Regular fluctuations that repeat at fixed intervals (like higher sales during holidays)
/// 3. Cyclical patterns: Longer-term ups and downs (like business cycles)
/// 4. Irregular components: Random fluctuations that don't follow any pattern
/// 
/// SEATS helps separate these components by:
/// 
/// 1. Modeling the time series using ARIMA (AutoRegressive Integrated Moving Average) methods
/// 2. Identifying and extracting the seasonal patterns
/// 3. Separating the trend from the irregular components
/// 
/// Why is SEATS important in AI and machine learning?
/// 
/// 1. Improved Forecasting: By understanding each component separately, predictions become more accurate
/// 
/// 2. Pattern Recognition: Helps AI systems identify meaningful patterns versus random noise
/// 
/// 3. Anomaly Detection: Makes it easier to spot unusual events that don't fit established patterns
/// 
/// 4. Feature Engineering: Creates useful features for machine learning models from time series data
/// 
/// 5. Seasonal Adjustment: Allows for fair comparisons between different time periods by removing seasonal effects
/// 
/// This enum specifies which specific algorithm variant to use for SEATS decomposition, as different methods have 
/// different characteristics and may be more suitable for certain types of time series data.
/// </para>
/// </remarks>
public enum SEATSAlgorithmType
{
    /// <summary>
    /// Uses the standard SEATS algorithm for time series decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Standard SEATS algorithm is the original implementation of the SEATS methodology, 
    /// developed by the Bank of Spain and widely used in official statistics.
    /// 
    /// The Standard approach:
    /// 
    /// 1. First fits an ARIMA model to your time series data (this is a statistical model that captures 
    ///    patterns in the data)
    /// 
    /// 2. Then uses signal extraction techniques based on this model to separate the series into components
    /// 
    /// 3. Works in the frequency domain (which means it analyzes the data in terms of cycles and frequencies)
    /// 
    /// 4. Ensures that the components add up exactly to the original series
    /// 
    /// 5. Produces components with well-defined statistical properties
    /// 
    /// This method is particularly good at:
    /// 
    /// 1. Handling complex seasonal patterns
    /// 
    /// 2. Providing theoretically sound decompositions
    /// 
    /// 3. Working with economic and financial time series
    /// 
    /// 4. Producing results that are consistent with economic theory
    /// 
    /// In machine learning applications, the Standard SEATS algorithm provides reliable seasonal adjustments 
    /// that can be used as pre-processing steps before training forecasting models, or for creating features 
    /// that capture the underlying trend in the data.
    /// </para>
    /// </remarks>
    Standard,

    /// <summary>
    /// Uses the canonical SEATS decomposition approach, which enforces specific constraints on the components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Canonical SEATS approach is a variation that imposes additional mathematical constraints 
    /// to ensure the decomposition has certain desirable properties.
    /// 
    /// Think of it like solving a puzzle with extra rules: the canonical approach adds specific requirements about 
    /// how the pieces (components) should fit together, making the solution more structured.
    /// 
    /// The Canonical approach:
    /// 
    /// 1. Enforces orthogonality between components (meaning the different components don't "overlap" statistically)
    /// 
    /// 2. Minimizes the variance of the irregular component (making the trend and seasonal components explain as 
    ///    much of the data as possible)
    /// 
    /// 3. Produces a unique decomposition (there's only one "right answer" given the constraints)
    /// 
    /// 4. Often results in smoother trend components
    /// 
    /// 5. Makes stronger assumptions about the statistical properties of the components
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You need components that are statistically independent from each other
    /// 
    /// 2. You want to maximize the explanatory power of your trend and seasonal components
    /// 
    /// 3. You're using the decomposition results as inputs to other statistical models
    /// 
    /// 4. You need a decomposition with well-defined mathematical properties
    /// 
    /// In machine learning contexts, the Canonical approach can provide cleaner input features for predictive models, 
    /// as the separation between components is more distinct, potentially leading to better model performance when 
    /// the components are used as features.
    /// </para>
    /// </remarks>
    Canonical,

    /// <summary>
    /// Uses Burman's variant of the SEATS algorithm, which focuses on robust seasonal adjustment.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Burman approach is a modified version of SEATS that emphasizes robustness and 
    /// practical applicability, especially for data with outliers or structural changes.
    /// 
    /// Imagine you're trying to find patterns in weather data that includes some extreme events like hurricanes. 
    /// The Burman approach is designed to handle these unusual observations better and still extract meaningful 
    /// seasonal patterns.
    /// 
    /// The Burman approach:
    /// 
    /// 1. Incorporates additional techniques to handle outliers and structural breaks in the data
    /// 
    /// 2. Often uses modified filter designs that are less sensitive to unusual observations
    /// 
    /// 3. May include pre-treatment of the data to identify and adjust for special events
    /// 
    /// 4. Focuses on practical seasonal adjustment rather than theoretical optimality
    /// 
    /// 5. Can adapt to changing seasonal patterns over time
    /// 
    /// This method is particularly valuable when:
    /// 
    /// 1. Your data contains outliers or anomalies
    /// 
    /// 2. The seasonal patterns change over time (evolving seasonality)
    /// 
    /// 3. You're working with real-world messy data rather than idealized time series
    /// 
    /// 4. Practical results are more important than theoretical purity
    /// 
    /// In machine learning applications, the Burman approach can provide more reliable seasonal adjustments for 
    /// real-world data, which is especially important when building models that need to be robust against outliers 
    /// or when analyzing data from volatile environments where patterns may not be perfectly stable.
    /// </para>
    /// </remarks>
    Burman
}
