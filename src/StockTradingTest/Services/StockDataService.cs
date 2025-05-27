using CsvHelper;
using CsvHelper.Configuration;
using StockTradingTest.Configuration;
using StockTradingTest.Models;
using System.Globalization;
using AiDotNet.LinearAlgebra;

namespace StockTradingTest.Services
{
    public class StockDataService
    {
        private readonly DataSourceConfig _config;
        private Dictionary<string, List<StockData>> _stockData = new Dictionary<string, List<StockData>>();

        public IReadOnlyDictionary<string, List<StockData>> StockData => _stockData;
        public IReadOnlyList<string> Symbols => _stockData.Keys.ToList();

        public StockDataService(DataSourceConfig config)
        {
            _config = config;
        }

        public async Task LoadDataAsync()
        {
            _stockData.Clear();
            
            foreach (var symbol in _config.DefaultSymbols)
            {
                string filePath = Path.Combine(_config.StockDataPath, $"{symbol}.csv");
                if (!File.Exists(filePath))
                {
                    await DownloadStockDataAsync(symbol);
                }

                _stockData[symbol] = await ReadStockDataFromCsvAsync(symbol);
            }
        }

        private async Task<List<StockData>> ReadStockDataFromCsvAsync(string symbol)
        {
            string filePath = Path.Combine(_config.StockDataPath, $"{symbol}.csv");
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                MissingFieldFound = null
            };

            using var reader = new StreamReader(filePath);
            using var csv = new CsvReader(reader, config);
            
            var records = new List<StockData>();
            
            await Task.Run(() => {
                csv.Read();
                csv.ReadHeader();
                
                while (csv.Read())
                {
                    var date = csv.GetField<DateTime>("Date");
                    
                    if (date < _config.StartDate || date > _config.EndDate)
                        continue;
                        
                    var record = new StockData
                    {
                        Symbol = symbol,
                        Date = date,
                        Open = csv.GetField<decimal>("Open"),
                        High = csv.GetField<decimal>("High"),
                        Low = csv.GetField<decimal>("Low"),
                        Close = csv.GetField<decimal>("Close"),
                        AdjustedClose = csv.TryGetField<decimal>("Adj Close", out var adjClose) ? adjClose : csv.GetField<decimal>("Close"),
                        Volume = csv.GetField<long>("Volume")
                    };
                    
                    records.Add(record);
                }
            });
            
            return records.OrderBy(r => r.Date).ToList();
        }

        private async Task DownloadStockDataAsync(string symbol)
        {
            // In a real implementation, this would download data from a financial API
            // For this example, we'll create a directory and throw an error that the data needs to be downloaded
            
            if (!Directory.Exists(_config.StockDataPath))
            {
                Directory.CreateDirectory(_config.StockDataPath);
            }
            
            throw new FileNotFoundException(
                $"Stock data for {symbol} not found. Please download historical data for this symbol and save it to {_config.StockDataPath}/{symbol}.csv");
        }

        public (Matrix<double> Features, Vector<double> Targets) PrepareModelData(
            string symbol, 
            DateTime startDate, 
            DateTime endDate,
            int lookbackPeriod, 
            int predictionHorizon,
            bool includeTargets = true)
        {
            if (!_stockData.ContainsKey(symbol))
            {
                throw new ArgumentException($"Data for symbol {symbol} not available");
            }

            var symbolData = _stockData[symbol];
            var filteredData = symbolData
                .Where(d => d.Date >= startDate && d.Date <= endDate)
                .OrderBy(d => d.Date)
                .ToList();

            if (filteredData.Count < lookbackPeriod + predictionHorizon)
            {
                throw new ArgumentException($"Not enough data for symbol {symbol} in the specified date range");
            }

            int numSamples = filteredData.Count - lookbackPeriod - (includeTargets ? predictionHorizon : 0);
            if (numSamples <= 0)
            {
                throw new ArgumentException("Not enough data to create samples with the specified lookback and prediction periods");
            }

            int numFeatures = _config.FeaturesToInclude.Count * lookbackPeriod;
            var features = new Matrix(numSamples, numFeatures);
            var targets = includeTargets ? new Vector(numSamples) : new Vector(0);

            for (int i = 0; i < numSamples; i++)
            {
                int featureIdx = 0;
                for (int j = 0; j < lookbackPeriod; j++)
                {
                    var dataPoint = filteredData[i + j];
                    foreach (var feature in _config.FeaturesToInclude)
                    {
                        decimal value = GetFeatureValue(dataPoint, feature);
                        features[i, featureIdx++] = (double)value;
                    }
                }

                if (includeTargets)
                {
                    // Target is the close price change after predictionHorizon days
                    decimal currentClose = filteredData[i + lookbackPeriod - 1].Close;
                    decimal futureClose = filteredData[i + lookbackPeriod + predictionHorizon - 1].Close;
                    targets[i] = (double)((futureClose / currentClose) - 1m);
                }
            }

            return (features, targets);
        }

        private decimal GetFeatureValue(StockData data, string featureName)
        {
            return featureName switch
            {
                "Open" => data.Open,
                "High" => data.High,
                "Low" => data.Low,
                "Close" => data.Close,
                "Volume" => data.Volume,
                "AdjustedClose" => data.AdjustedClose,
                _ => throw new ArgumentException($"Unknown feature: {featureName}")
            };
        }

        public List<DateTime> GetTradingDays(DateTime startDate, DateTime endDate, string symbol)
        {
            if (!_stockData.ContainsKey(symbol))
            {
                throw new ArgumentException($"Data for symbol {symbol} not available");
            }

            return _stockData[symbol]
                .Where(d => d.Date >= startDate && d.Date <= endDate)
                .Select(d => d.Date)
                .OrderBy(d => d)
                .ToList();
        }

        public StockData? GetStockData(string symbol, DateTime date)
        {
            if (!_stockData.ContainsKey(symbol))
            {
                return null;
            }

            return _stockData[symbol].FirstOrDefault(d => d.Date.Date == date.Date);
        }

        public Dictionary<string, StockData?> GetStockDataForAllSymbols(DateTime date)
        {
            return _stockData.Keys.ToDictionary(
                symbol => symbol,
                symbol => GetStockData(symbol, date)
            );
        }
    }
}