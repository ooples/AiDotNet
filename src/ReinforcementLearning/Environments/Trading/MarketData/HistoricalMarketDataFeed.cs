using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.MarketData
{
    /// <summary>
    /// A market data feed that uses historical data from CSV files or other sources.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class HistoricalMarketDataFeed<T> : IMarketDataFeed<T>
    {
        private readonly List<MarketDataPoint<T>> _data = default!;
        private readonly Random _random = default!;
        private readonly int _warmupPeriod;

        /// <summary>
        /// Gets the current position in the data feed.
        /// </summary>
        public int CurrentPosition { get; private set; }

        /// <summary>
        /// Gets the total number of available data points.
        /// </summary>
        public int Length => _data.Count;

        /// <summary>
        /// Gets the assets/symbols available in this data feed.
        /// </summary>
        public IEnumerable<string> Symbols { get; }

        /// <summary>
        /// Gets the features available for each symbol.
        /// </summary>
        public IEnumerable<string> Features { get; }

        /// <summary>
        /// Gets the timestamp for the current data point.
        /// </summary>
        public DateTime CurrentTimestamp => _data[CurrentPosition].Timestamp;

        /// <summary>
        /// Initializes a new instance of the <see cref="HistoricalMarketDataFeed{T}"/> class.
        /// </summary>
        /// <param name="data">The historical market data.</param>
        /// <param name="warmupPeriod">The number of data points to reserve at the beginning for warmup.</param>
        public HistoricalMarketDataFeed(List<MarketDataPoint<T>> data, int warmupPeriod = 0)
        {
            _data = data ?? throw new ArgumentNullException(nameof(data));
            _random = new Random();
            _warmupPeriod = warmupPeriod;

            if (_data.Count == 0)
            {
                throw new ArgumentException("Data cannot be empty.");
            }

            // Extract symbols and features from the first data point
            Symbols = _data[0].Symbols.ToList();
            Features = _data[0].Features.ToList();

            // Initialize position to after warmup period
            CurrentPosition = warmupPeriod;
        }

        /// <summary>
        /// Gets the available data for the current time step.
        /// </summary>
        /// <returns>A market data point containing all available data.</returns>
        public MarketDataPoint<T> GetCurrentData()
        {
            if (CurrentPosition < 0 || CurrentPosition >= _data.Count)
            {
                throw new InvalidOperationException("Current position is out of range.");
            }

            return _data[CurrentPosition];
        }

        /// <summary>
        /// Gets the available data for a specific time step.
        /// </summary>
        /// <param name="index">The time step index.</param>
        /// <returns>A market data point containing all available data.</returns>
        public MarketDataPoint<T> GetData(int index)
        {
            if (index < 0 || index >= _data.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range.");
            }

            return _data[index];
        }

        /// <summary>
        /// Advances the data feed to the next time step.
        /// </summary>
        /// <returns>True if successfully advanced, false if at the end of the data.</returns>
        public bool MoveNext()
        {
            if (CurrentPosition < _data.Count - 1)
            {
                CurrentPosition++;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Moves the data feed to a specific time step.
        /// </summary>
        /// <param name="index">The time step index to move to.</param>
        /// <returns>True if successfully moved, false otherwise.</returns>
        public bool MoveTo(int index)
        {
            if (index < _warmupPeriod || index >= _data.Count)
            {
                return false;
            }

            CurrentPosition = index;
            return true;
        }

        /// <summary>
        /// Resets the data feed to the beginning or a random valid position for training.
        /// </summary>
        /// <param name="random">Whether to reset to a random position (useful for training).</param>
        public void Reset(bool random = false)
        {
            if (random)
            {
                // Reset to a random position after warmup period
                CurrentPosition = _random.Next(_warmupPeriod, _data.Count);
            }
            else
            {
                // Reset to just after warmup period
                CurrentPosition = _warmupPeriod;
            }
        }

        /// <summary>
        /// Gets a specific feature for all symbols at the current time step.
        /// </summary>
        /// <param name="feature">The feature name (e.g., "Close").</param>
        /// <returns>A dictionary mapping each symbol to its feature value.</returns>
        public Dictionary<string, T> GetFeatureForAllSymbols(string feature)
        {
            return _data[CurrentPosition].GetAllValuesForFeature(feature);
        }

        /// <summary>
        /// Gets historical values of a specific feature for a specific symbol.
        /// </summary>
        /// <param name="symbol">The market symbol.</param>
        /// <param name="feature">The feature name.</param>
        /// <param name="lookback">The number of historical values to include.</param>
        /// <returns>An array of historical values.</returns>
        public T[] GetHistoricalValues(string symbol, string feature, int lookback)
        {
            if (lookback <= 0)
            {
                throw new ArgumentException("Lookback must be positive.", nameof(lookback));
            }

            if (CurrentPosition < lookback)
            {
                throw new InvalidOperationException($"Not enough historical data available. Current position: {CurrentPosition}, Lookback: {lookback}");
            }

            var result = new T[lookback];
            for (int i = 0; i < lookback; i++)
            {
                int index = CurrentPosition - lookback + i;
                result[i] = _data[index].GetValue(symbol, feature);
            }

            return result;
        }

        /// <summary>
        /// Creates a historical market data feed from CSV files.
        /// </summary>
        /// <param name="filePaths">Dictionary mapping symbol names to their CSV file paths.</param>
        /// <param name="dateFormat">The date format used in the CSV files.</param>
        /// <param name="dateColumn">The name of the column containing dates.</param>
        /// <param name="columnMappings">Dictionary mapping CSV column names to standardized feature names.</param>
        /// <param name="warmupPeriod">The number of data points to reserve at the beginning for warmup.</param>
        /// <returns>A historical market data feed.</returns>
        /// <exception cref="ArgumentException">Thrown when no files are provided or files have different date ranges.</exception>
        public static HistoricalMarketDataFeed<T> FromCSV(
            Dictionary<string, string> filePaths,
            string dateFormat = "yyyy-MM-dd",
            string dateColumn = "Date",
            Dictionary<string, string>? columnMappings = null,
            int warmupPeriod = 20)
        {
            if (filePaths == null || filePaths.Count == 0)
            {
                throw new ArgumentException("No file paths provided.");
            }

            // Dictionary to store all parsed data by date
            var dataByDate = new Dictionary<DateTime, Dictionary<string, Dictionary<string, T>>>();
            
            // If no column mappings provided, use default OHLCV mappings
            columnMappings ??= new Dictionary<string, string>
            {
                { "Open", "Open" },
                { "High", "High" },
                { "Low", "Low" },
                { "Close", "Close" },
                { "Volume", "Volume" }
            };

            // Parse each CSV file
            foreach (var symbolFile in filePaths)
            {
                string symbol = symbolFile.Key;
                string filePath = symbolFile.Value;

                try
                {
                    // Read all lines from the CSV file
                    string[] lines = File.ReadAllLines(filePath);
                    if (lines.Length <= 1)
                    {
                        throw new ArgumentException($"File {filePath} has no data rows.");
                    }

                    // Parse header to find column indices
                    string[] headers = lines[0].Split(',');
                    var columnIndices = new Dictionary<string, int>();
                    
                    int dateColumnIndex = -1;
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string header = headers[i].Trim();
                        if (header.Equals(dateColumn, StringComparison.OrdinalIgnoreCase))
                        {
                            dateColumnIndex = i;
                        }
                        else
                        {
                            // For each column that has a mapping, store its index
                            foreach (var mapping in columnMappings)
                            {
                                if (header.Equals(mapping.Key, StringComparison.OrdinalIgnoreCase))
                                {
                                    columnIndices[mapping.Value] = i;
                                    break;
                                }
                            }
                        }
                    }

                    if (dateColumnIndex == -1)
                    {
                        throw new ArgumentException($"Date column '{dateColumn}' not found in file {filePath}.");
                    }

                    // Parse each data row
                    for (int i = 1; i < lines.Length; i++)
                    {
                        string[] values = lines[i].Split(',');
                        if (values.Length <= dateColumnIndex)
                        {
                            continue; // Skip malformed rows
                        }

                        // Parse date
                        if (!DateTime.TryParseExact(values[dateColumnIndex].Trim(), dateFormat, null, System.Globalization.DateTimeStyles.None, out DateTime date))
                        {
                            continue; // Skip rows with invalid dates
                        }

                        // Create or get the data dictionary for this date
                        if (!dataByDate.TryGetValue(date, out var dateData))
                        {
                            dateData = new Dictionary<string, Dictionary<string, T>>();
                            dataByDate[date] = dateData;
                        }

                        // Create feature dictionary for this symbol
                        var featureDict = new Dictionary<string, T>();
                        
                        // Parse each mapped column
                        foreach (var columnEntry in columnIndices)
                        {
                            string feature = columnEntry.Key;
                            int columnIndex = columnEntry.Value;

                            if (columnIndex < values.Length && double.TryParse(values[columnIndex].Trim(), out double value))
                            {
                                // Convert to appropriate numeric type
                                featureDict[feature] = (T)Convert.ChangeType(value, typeof(T));
                            }
                        }

                        // Add this symbol's data to the date data
                        dateData[symbol] = featureDict;
                    }
                }
                catch (Exception ex)
                {
                    throw new Exception($"Error parsing file {filePath}: {ex.Message}", ex);
                }
            }

            // Convert the data dictionary to a chronologically sorted list of data points
            var sortedDates = dataByDate.Keys.OrderBy(d => d).ToList();
            var dataPoints = new List<MarketDataPoint<T>>();

            foreach (var date in sortedDates)
            {
                var dateData = dataByDate[date];
                
                // Only include dates that have data for all symbols
                if (dateData.Count == filePaths.Count)
                {
                    dataPoints.Add(new MarketDataPoint<T>(date, dateData));
                }
            }

            if (dataPoints.Count == 0)
            {
                throw new ArgumentException("No valid data points found after processing all files.");
            }

            return new HistoricalMarketDataFeed<T>(dataPoints, warmupPeriod);
        }
    }
}