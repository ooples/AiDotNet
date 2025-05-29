using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.MarketData
{
    /// <summary>
    /// Represents a single point of market data containing price and other information for all symbols at a specific time.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MarketDataPoint<T>
    {
        /// <summary>
        /// Gets the timestamp for this data point.
        /// </summary>
        public DateTime Timestamp { get; }

        /// <summary>
        /// Gets the data dictionary containing symbol -> feature -> value mapping.
        /// </summary>
        public Dictionary<string, Dictionary<string, T>> Data { get; }

        /// <summary>
        /// Gets the available symbols in this data point.
        /// </summary>
        public IEnumerable<string> Symbols => Data.Keys;

        /// <summary>
        /// Gets all available features across all symbols.
        /// </summary>
        public IEnumerable<string> Features => 
            Data.Values
                .SelectMany(featureDict => featureDict.Keys)
                .Distinct();

        /// <summary>
        /// Initializes a new instance of the <see cref="MarketDataPoint{T}"/> class.
        /// </summary>
        /// <param name="timestamp">The timestamp for this data point.</param>
        /// <param name="data">The data dictionary containing symbol -> feature -> value mapping.</param>
        public MarketDataPoint(DateTime timestamp, Dictionary<string, Dictionary<string, T>> data)
        {
            Timestamp = timestamp;
            Data = data;
        }

        /// <summary>
        /// Gets the value of a specific feature for a specific symbol.
        /// </summary>
        /// <param name="symbol">The symbol to get data for.</param>
        /// <param name="feature">The feature to get.</param>
        /// <returns>The feature value.</returns>
        /// <exception cref="KeyNotFoundException">Thrown when the symbol or feature is not found.</exception>
        public T GetValue(string symbol, string feature)
        {
            if (!Data.TryGetValue(symbol, out var featureDict))
            {
                throw new KeyNotFoundException($"Symbol '{symbol}' not found in market data.");
            }

            if (!featureDict.TryGetValue(feature, out var value))
            {
                throw new KeyNotFoundException($"Feature '{feature}' not found for symbol '{symbol}'.");
            }

            return value;
        }

        /// <summary>
        /// Tries to get the value of a specific feature for a specific symbol.
        /// </summary>
        /// <param name="symbol">The symbol to get data for.</param>
        /// <param name="feature">The feature to get.</param>
        /// <param name="value">The feature value if found.</param>
        /// <returns>True if the value was found, false otherwise.</returns>
        public bool TryGetValue(string symbol, string feature, out T value)
        {
            if (!Data.TryGetValue(symbol, out var featureDict))
            {
                value = default(T)!;
                return false;
            }

            if (featureDict.TryGetValue(feature, out var tempValue))
            {
                value = tempValue;
                return true;
            }
            
            value = default(T)!;
            return false;
        }

        /// <summary>
        /// Gets all values for a specific feature across all symbols.
        /// </summary>
        /// <param name="feature">The feature to get.</param>
        /// <returns>A dictionary mapping symbols to their feature values.</returns>
        public Dictionary<string, T> GetAllValuesForFeature(string feature)
        {
            var result = new Dictionary<string, T>();

            foreach (var symbolData in Data)
            {
                if (symbolData.Value.TryGetValue(feature, out var value))
                {
                    result[symbolData.Key] = value;
                }
            }

            return result;
        }

        /// <summary>
        /// Gets all feature values for a specific symbol.
        /// </summary>
        /// <param name="symbol">The symbol to get data for.</param>
        /// <returns>A dictionary mapping features to their values.</returns>
        /// <exception cref="KeyNotFoundException">Thrown when the symbol is not found.</exception>
        public Dictionary<string, T> GetAllFeaturesForSymbol(string symbol)
        {
            if (!Data.TryGetValue(symbol, out var featureDict))
            {
                throw new KeyNotFoundException($"Symbol '{symbol}' not found in market data.");
            }

            return new Dictionary<string, T>(featureDict);
        }
    }
}