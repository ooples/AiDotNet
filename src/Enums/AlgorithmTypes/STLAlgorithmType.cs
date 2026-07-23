namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Seasonal-Trend decomposition using LOESS (STL).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> STL (Seasonal-Trend decomposition using LOESS) is a technique that breaks down time series data 
/// into three components: seasonal patterns, trend, and remainder (or residual).
/// 
/// Imagine you're analyzing monthly ice cream sales over several years:
/// 
/// 1. Seasonal Component: This captures regular patterns that repeat at fixed intervals. For ice cream sales, 
///    this would show higher sales in summer months and lower sales in winter months, repeating each year.
/// 
/// 2. Trend Component: This represents the long-term progression of your data, ignoring seasonality and noise. 
///    For ice cream sales, this might show a gradual increase over the years as your business grows.
/// 
/// 3. Remainder Component: This contains what's left after removing the seasonal and trend components. It 
///    represents irregular fluctuations, random noise, or unusual events (like a sudden spike in sales during 
///    an unexpected heat wave).
/// 
/// STL uses a method called LOESS (Locally Estimated Scatterplot Smoothing) to perform this decomposition. 
/// LOESS works by fitting simple models to small chunks of the data at a time, which makes it flexible and 
/// able to capture complex patterns.
/// 
/// Why is STL important in AI and machine learning?
/// 
/// 1. Feature Engineering: The components can be used as separate features in machine learning models
/// 
/// 2. Forecasting: Understanding seasonal patterns and trends helps make better predictions
/// 
/// 3. Anomaly Detection: Unusual values in the remainder component can indicate anomalies
/// 
/// 4. Data Preprocessing: Removing seasonality can help models focus on underlying patterns
/// 
/// 5. Interpretability: Breaking down complex time series makes the data more understandable
/// 
/// This enum specifies which specific algorithm variant to use for STL decomposition, as different methods 
/// have different strengths and may be more suitable for certain types of data or analysis goals.
/// </para>
/// </remarks>
public enum STLAlgorithmType
{
    /// <summary>
    /// Uses the standard implementation of the STL algorithm for time series decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Standard STL algorithm is the original implementation that follows the classic approach 
    /// described by Cleveland et al. (1990).
    /// 
    /// Think of it as the "classic recipe" for STL decomposition:
    /// 
    /// 1. It uses nested loops (inner and outer loops) to iteratively refine the decomposition
    /// 
    /// 2. The inner loop focuses on separating the seasonal component from the trend
    /// 
    /// 3. The outer loop helps identify and reduce the impact of outliers
    /// 
    /// 4. It applies LOESS smoothing at multiple steps to extract smooth seasonal and trend components
    /// 
    /// The Standard approach:
    /// 
    /// 1. Is well-tested and widely used in statistical analysis
    /// 
    /// 2. Produces high-quality decompositions for most well-behaved time series
    /// 
    /// 3. Handles a wide range of seasonal patterns
    /// 
    /// 4. Provides a good balance between accuracy and computational efficiency
    /// 
    /// 5. Has well-understood statistical properties
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You want a reliable, proven approach to time series decomposition
    /// 
    /// 2. Your data has clear seasonal patterns
    /// 
    /// 3. You need a method that works well across many different types of time series
    /// 
    /// 4. You're looking for a good default choice for time series analysis
    /// 
    /// In machine learning applications, the Standard STL algorithm provides reliable decompositions that can 
    /// improve forecasting models by allowing them to learn from the trend and seasonal components separately.
    /// </para>
    /// </remarks>
    Standard,

    /// <summary>
    /// Uses a robust version of the STL algorithm that is less sensitive to outliers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Robust STL algorithm is a variation designed to handle data with outliers or unusual 
    /// observations without letting them distort the overall decomposition.
    /// 
    /// Imagine you're trying to understand the pattern of daily website traffic, but occasionally there are huge 
    /// spikes due to viral content. The Robust approach prevents these rare events from skewing your understanding 
    /// of the normal patterns:
    /// 
    /// 1. It uses weight functions that give less importance to outlier values
    /// 
    /// 2. It iteratively identifies and downweights unusual observations
    /// 
    /// 3. It focuses on capturing the patterns in the majority of the data
    /// 
    /// The Robust approach:
    /// 
    /// 1. Is more resistant to the influence of outliers and anomalies
    /// 
    /// 2. Produces more stable seasonal and trend components when data contains unusual observations
    /// 
    /// 3. May require more computational resources than the standard approach
    /// 
    /// 4. Often produces cleaner decompositions for real-world, messy data
    /// 
    /// 5. Helps identify outliers by examining points that receive low weights
    /// 
    /// This method is particularly valuable when:
    /// 
    /// 1. Your data contains outliers or anomalies
    /// 
    /// 2. You're working with noisy real-world data
    /// 
    /// 3. You want to prevent unusual events from distorting your understanding of normal patterns
    /// 
    /// 4. The quality of the decomposition is more important than computational speed
    /// 
    /// In machine learning applications, the Robust STL algorithm can provide cleaner input features for predictive 
    /// models by preventing outliers from corrupting the seasonal and trend components, leading to more reliable 
    /// forecasts and pattern recognition.
    /// </para>
    /// </remarks>
    Robust,

    /// <summary>
    /// Uses an optimized version of the STL algorithm designed for speed and efficiency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Fast STL algorithm is an optimized version that sacrifices some precision for 
    /// significant gains in computational speed, making it suitable for large datasets or real-time applications.
    /// 
    /// Think of it like using a blender instead of chopping vegetables by hand - you might lose some control 
    /// over the exact size of the pieces, but you'll finish much faster:
    /// 
    /// 1. It may use fewer iterations or simplified calculations
    /// 
    /// 2. It might employ approximation techniques to speed up computations
    /// 
    /// 3. It could use more efficient data structures or parallel processing
    /// 
    /// The Fast approach:
    /// 
    /// 1. Is significantly faster than the standard or robust approaches
    /// 
    /// 2. Requires less computational resources
    /// 
    /// 3. May sacrifice some accuracy or detail in the decomposition
    /// 
    /// 4. Is suitable for very large time series or real-time processing
    /// 
    /// 5. Often provides "good enough" results for many practical applications
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You're working with very large datasets
    /// 
    /// 2. You need real-time or near-real-time processing
    /// 
    /// 3. Computational resources are limited
    /// 
    /// 4. You're doing exploratory analysis and need quick results
    /// 
    /// 5. The slight loss in precision is acceptable for your application
    /// 
    /// In machine learning applications, the Fast STL algorithm enables efficient processing of large-scale time 
    /// series data, making it practical to incorporate time series decomposition into production systems or to 
    /// quickly analyze multiple time series during model development and feature engineering.
    /// </para>
    /// </remarks>
    Fast
}
