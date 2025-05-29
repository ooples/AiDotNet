using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Environments.Trading.MarketData;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Represents a financial market data feed that provides price and other market information.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IMarketDataFeed<T>
    {
        /// <summary>
        /// Gets the available data for the current time step.
        /// </summary>
        /// <returns>A dictionary containing market data values.</returns>
        MarketDataPoint<T> GetCurrentData();

        /// <summary>
        /// Gets the available data for a specific time step.
        /// </summary>
        /// <param name="index">The time step index.</param>
        /// <returns>A dictionary containing market data values.</returns>
        MarketDataPoint<T> GetData(int index);

        /// <summary>
        /// Advances the data feed to the next time step.
        /// </summary>
        /// <returns>True if successfully advanced, false if at the end of the data.</returns>
        bool MoveNext();

        /// <summary>
        /// Moves the data feed to a specific time step.
        /// </summary>
        /// <param name="index">The time step index to move to.</param>
        /// <returns>True if successfully moved, false otherwise.</returns>
        bool MoveTo(int index);

        /// <summary>
        /// Resets the data feed to the beginning or a random valid position for training.
        /// </summary>
        /// <param name="random">Whether to reset to a random position (useful for training).</param>
        void Reset(bool random = false);

        /// <summary>
        /// Gets the total number of available data points.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Gets the current position in the data feed.
        /// </summary>
        int CurrentPosition { get; }

        /// <summary>
        /// Gets the assets/symbols available in this data feed.
        /// </summary>
        IEnumerable<string> Symbols { get; }

        /// <summary>
        /// Gets the features available for each symbol (like Open, High, Low, Close, Volume, etc.).
        /// </summary>
        IEnumerable<string> Features { get; }

        /// <summary>
        /// Gets a specific feature for all symbols at the current time step.
        /// </summary>
        /// <param name="feature">The feature name (e.g., "Close").</param>
        /// <returns>A dictionary mapping each symbol to its feature value.</returns>
        Dictionary<string, T> GetFeatureForAllSymbols(string feature);

        /// <summary>
        /// Gets historical values of a specific feature for a specific symbol.
        /// </summary>
        /// <param name="symbol">The market symbol.</param>
        /// <param name="feature">The feature name.</param>
        /// <param name="lookback">The number of historical values to include.</param>
        /// <returns>An array of historical values.</returns>
        T[] GetHistoricalValues(string symbol, string feature, int lookback);

        /// <summary>
        /// Gets the timestamp for the current data point.
        /// </summary>
        DateTime CurrentTimestamp { get; }
    }
}