namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods and properties for decomposing time series data into its component parts.
/// </summary>
/// <remarks>
/// Time series decomposition breaks down a sequence of data points into underlying patterns
/// such as trend, seasonality, and residual components.
/// 
/// <b>For Beginners:</b> Time series decomposition is like taking apart a complex signal (like sales data
/// over time) into simpler pieces that are easier to understand. Imagine your store's sales
/// throughout the year - decomposition helps you separate:
/// 
/// - The overall growth trend (are sales generally increasing?)
/// - Seasonal patterns (higher sales during holidays?)
/// - Day-to-day random variations
/// 
/// This makes it easier to understand what's really happening in your data and make better predictions.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("TimeSeriesDecomposition")]
public interface ITimeSeriesDecomposition<T>
{
    /// <summary>
    /// Gets the original time series data that was decomposed.
    /// </summary>
    /// <remarks>
    /// This property provides access to the input data used for the decomposition.
    /// 
    /// <b>For Beginners:</b> This is simply your original data - the sequence of values over time
    /// that you wanted to analyze. For example, if you're analyzing daily temperature readings,
    /// this would be your complete list of temperature values in chronological order.
    /// </remarks>
    Vector<T> TimeSeries { get; }

    /// <summary>
    /// Gets all available decomposition components as a dictionary.
    /// </summary>
    /// <remarks>
    /// This method returns all the extracted components from the time series.
    /// 
    /// <b>For Beginners:</b> This gives you all the pieces that make up your data, organized by type.
    /// Common components include:
    /// 
    /// - Trend: The long-term direction (going up, down, or staying flat)
    /// - Seasonality: Regular patterns that repeat (like higher sales every weekend)
    /// - Cycle: Longer patterns that don't have a fixed frequency
    /// - Residual/Noise: The random variations that don't fit any pattern
    /// 
    /// Each component helps you understand a different aspect of what's happening in your data.
    /// </remarks>
    /// <returns>A dictionary mapping component types to their corresponding data.</returns>
    Dictionary<DecompositionComponentType, object> GetComponents();

    /// <summary>
    /// Gets a specific component of the time series decomposition.
    /// </summary>
    /// <remarks>
    /// This method retrieves a single component by its type.
    /// 
    /// <b>For Beginners:</b> This lets you look at just one specific pattern in your data. For example,
    /// if you only want to see the seasonal pattern in your sales data (like how much sales
    /// increase during holidays), you would use this method to get just that piece.
    /// 
    /// The component is returned as an "object" type, which means you'll need to convert it
    /// to the appropriate type (usually a Vector&lt;T&gt;) before using it.
    /// </remarks>
    /// <param name="componentType">The type of component to retrieve (e.g., Trend, Seasonal, Residual).</param>
    /// <returns>The requested component data, or null if the component doesn't exist.</returns>
    object? GetComponent(DecompositionComponentType componentType);

    /// <summary>
    /// Checks if a specific decomposition component is available.
    /// </summary>
    /// <remarks>
    /// This method determines whether a particular component was extracted during decomposition.
    /// 
    /// <b>For Beginners:</b> Not all decomposition methods extract all possible components. This method
    /// lets you check if a specific component is available before trying to use it.
    /// 
    /// For example, some simple decomposition methods might not separate cyclical patterns
    /// from the trend. You can use this method to check if the "Cycle" component exists
    /// before trying to analyze it.
    /// </remarks>
    /// <param name="componentType">The type of component to check for (e.g., Trend, Seasonal, Residual).</param>
    /// <returns>True if the component exists; otherwise, false.</returns>
    bool HasComponent(DecompositionComponentType componentType);
}
