using StockTradingTest.Configuration;
using StockTradingTest.Services;
using StockTradingTest.Models;

namespace StockTradingTest.Examples;

public class ProductionExample
{
    private static IFullModel<double, Vector<double>, Vector<double>>? _productionModel;
    private static MeanVarianceNormalizer<double, Vector<double>, Vector<double>> _normalizer = new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>();
    private static LoggingConfig _loggingConfig = new LoggingConfig();
    private static SimulationLogger _logger;
    private static TradingSimulationConfig _tradingConfig = new TradingSimulationConfig();
    private static ModelCompetitionConfig _modelConfig = new ModelCompetitionConfig();
    private static readonly Dictionary<string, List<StockData>> _recentData = new Dictionary<string, List<StockData>>();
    private static readonly List<Position> _openPositions = new List<Position>();
    private static decimal _availableCash;

    public static async Task RunExample()
    {
        Console.WriteLine("Starting Production Trading Example");
        
        // Load configuration
        var config = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .Build();

        config.GetSection("TradingSimulation").Bind(_tradingConfig);
        config.GetSection("ModelCompetition").Bind(_modelConfig);
        config.GetSection("Logging").Bind(_loggingConfig);

        _logger = new SimulationLogger(_loggingConfig);
        _logger.Log(LoggingLevel.Information, "Production Trading Example Started");

        // Set initial balance
        _availableCash = _tradingConfig.InitialBalance;

        try
        {
            // Load the pre-trained model
            await LoadProductionModel("BestTradingModel.json");
            
            if (_productionModel == null)
            {
                _logger.Log(LoggingLevel.Error, "Could not load production model. Running tournament first to create one.");
                
                // If no model exists, run the simple example to create one
                await SimpleUsageExample.RunExample();
                
                // Try loading again
                await LoadProductionModel("BestTradingModel.json");
                
                if (_productionModel == null)
                {
                    throw new Exception("Failed to load or create a production model");
                }
            }
            
            // Setup data retrieval (in a real system, this would connect to a broker API)
            await SetupDataFeed();
            
            // Show trading interface
            await ShowTradingInterface();
            
            _logger.Log(LoggingLevel.Information, "Production Trading Example Completed");
        }
        catch (Exception ex)
        {
            _logger.Log(LoggingLevel.Error, $"Error in production example: {ex.Message}");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
    
    private static async Task LoadProductionModel(string filePath)
    {
        _logger.Log(LoggingLevel.Information, $"Loading production model from {filePath}");
        
        if (!File.Exists(filePath))
        {
            _logger.Log(LoggingLevel.Warning, $"Model file {filePath} not found");
            return;
        }
        
        try
        {
            // In a real implementation, we would deserialize the model properly
            // For this example, we create a simple neural network
            
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new AiDotNet.Models.Options.NeuralNetworkRegressionOptions
            {
                Layers = new int[] { 30, 15, 1 },
                Epochs = 50,
                BatchSize = 32,
                LearningRate = 0.001,
                Optimizer = AiDotNet.Enums.OptimizerType.Adam,
                ActivationFunction = AiDotNet.Enums.ActivationFunction.ReLU
            });

            _productionModel = builder.BuildFullModel();
            
            // In a real system, we would load trained weights here
            _logger.Log(LoggingLevel.Information, "Production model loaded successfully");
            
            await Task.CompletedTask; // For async consistency
        }
        catch (Exception ex)
        {
            _logger.Log(LoggingLevel.Error, $"Error loading model: {ex.Message}");
            throw;
        }
    }
    
    private static async Task SetupDataFeed()
    {
        _logger.Log(LoggingLevel.Information, "Setting up data feed");
        
        // In a real system, this would connect to a broker's API
        // For this example, we'll create some synthetic data
        
        var symbols = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "META" };
        var startDate = DateTime.Now.AddDays(-_modelConfig.LookbackPeriod);
        var endDate = DateTime.Now;
        
        foreach (var symbol in symbols)
        {
            _recentData[symbol] = GenerateRecentStockData(symbol, startDate, endDate);
        }
        
        _logger.Log(LoggingLevel.Information, $"Data feed initialized with {symbols.Length} symbols");
        
        await Task.CompletedTask; // For async consistency
    }
    
    private static List<StockData> GenerateRecentStockData(string symbol, DateTime startDate, DateTime endDate)
    {
        var data = new List<StockData>();
        
        // Generate data with some randomness but a general trend
        Random rand = new Random(symbol.GetHashCode()); // Seed based on symbol
        
        // Initial price based on symbol hash
        double basePrice = 50.0 + (symbol.GetHashCode() % 20) * 10;
        double price = basePrice;
        double volatility = 0.01; // 1% daily volatility
        
        for (DateTime date = startDate; date <= endDate; date = date.AddDays(1))
        {
            // Skip weekends
            if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
                continue;
            
            // Calculate daily change
            double dailyReturn = rand.NextDouble() * volatility * 2 - volatility;
            price *= (1 + dailyReturn);
            
            // Calculate OHLC
            double open = price;
            double close = price * (1 + (rand.NextDouble() * 0.01 - 0.005));
            double high = Math.Max(open, close) * (1 + rand.NextDouble() * 0.01);
            double low = Math.Min(open, close) * (1 - rand.NextDouble() * 0.01);
            
            // Ensure minimum price
            if (low < 1) low = 1;
            if (open < low) open = low;
            if (close < low) close = low;
            if (high < Math.Max(open, close)) high = Math.Max(open, close) * 1.001;
            
            // Generate volume
            long volume = (long)(rand.NextDouble() * 10000000 + 500000);
            
            // Add the data
            data.Add(new StockData 
            {
                Symbol = symbol,
                Date = date,
                Open = (decimal)open,
                High = (decimal)high,
                Low = (decimal)low,
                Close = (decimal)close,
                AdjustedClose = (decimal)close,
                Volume = volume
            });
            
            // Update price for next day
            price = close;
        }
        
        return data;
    }
    
    private static async Task ShowTradingInterface()
    {
        bool running = true;
        
        while (running)
        {
            Console.Clear();
            Console.WriteLine("\n===== PRODUCTION TRADING INTERFACE =====\n");
            
            // Show portfolio summary
            PrintPortfolioSummary();
            
            // Show current positions
            PrintPositions();
            
            // Show recommended trades based on model predictions
            var recommendations = await GetTradeRecommendations();
            PrintTradeRecommendations(recommendations);
            
            // Show menu
            Console.WriteLine("\nMENU:");
            Console.WriteLine("1. Execute recommended trades");
            Console.WriteLine("2. Close a position");
            Console.WriteLine("3. Refresh market data");
            Console.WriteLine("4. Exit");
            
            Console.Write("\nEnter choice (1-4): ");
            var choice = Console.ReadLine()?.Trim();
            
            switch (choice)
            {
                case "1":
                    await ExecuteRecommendedTrades(recommendations);
                    break;
                case "2":
                    await ClosePosition();
                    break;
                case "3":
                    await RefreshMarketData();
                    break;
                case "4":
                    running = false;
                    break;
                default:
                    Console.WriteLine("Invalid choice. Press any key to continue...");
                    Console.ReadKey();
                    break;
            }
        }
    }
    
    private static void PrintPortfolioSummary()
    {
        decimal positionsValue = _openPositions.Sum(p => p.CurrentValue);
        decimal totalValue = _availableCash + positionsValue;
        decimal profitLoss = totalValue - _tradingConfig.InitialBalance;
        decimal profitLossPercent = (totalValue / _tradingConfig.InitialBalance) - 1.0m;
        
        Console.WriteLine($"Available Cash:     {_availableCash:C2}");
        Console.WriteLine($"Positions Value:    {positionsValue:C2}");
        Console.WriteLine($"Total Value:        {totalValue:C2}");
        Console.WriteLine($"Total P/L:          {profitLoss:C2} ({profitLossPercent:P2})");
        Console.WriteLine($"Open Positions:     {_openPositions.Count}/{_tradingConfig.MaxPositions}");
        Console.WriteLine();
    }
    
    private static void PrintPositions()
    {
        if (_openPositions.Count == 0)
        {
            Console.WriteLine("No open positions");
            Console.WriteLine();
            return;
        }
        
        Console.WriteLine("OPEN POSITIONS:");
        Console.WriteLine(new string('-', 80));
        Console.WriteLine($"{"Symbol",-8}{"Type",-8}{"Entry Price",-15}{"Current",-15}{"P/L %",-10}{"Value",-15}{"Entry Date",-12}");
        Console.WriteLine(new string('-', 80));
        
        foreach (var position in _openPositions)
        {
            string type = position.Type == PositionType.Long ? "LONG" : "SHORT";
            string plClass = position.ProfitLossPercent >= 0 ? "+" : "";
            
            Console.WriteLine(
                $"{position.Symbol,-8}{type,-8}{position.EntryPrice:C2,-15}{position.CurrentPrice:C2,-15}" +
                $"{plClass}{position.ProfitLossPercent:P2,-10}{position.CurrentValue:C2,-15}" +
                $"{position.EntryDate:MM/dd/yyyy,-12}");
        }
        
        Console.WriteLine();
    }
    
    private static async Task<List<(string Symbol, double PredictedReturn, string Recommendation)>> GetTradeRecommendations()
    {
        var recommendations = new List<(string Symbol, double PredictedReturn, string Recommendation)>();
        
        if (_productionModel == null)
        {
            return recommendations;
        }
        
        foreach (var symbol in _recentData.Keys)
        {
            // Skip if we already have a position
            if (_openPositions.Any(p => p.Symbol == symbol))
            {
                continue;
            }
            
            try
            {
                // Prepare features
                var recentData = _recentData[symbol].OrderByDescending(d => d.Date).Take(_modelConfig.LookbackPeriod).ToList();
                if (recentData.Count < _modelConfig.LookbackPeriod)
                {
                    continue;
                }
                
                // Create features - in a real system this would match the training data preparation
                var features = new Matrix(1, _modelConfig.LookbackPeriod * 5); // 5 features (OHLCV)
                int featureIdx = 0;
                
                // Reverse to get oldest data first
                foreach (var data in recentData.OrderBy(d => d.Date))
                {
                    features[0, featureIdx++] = (double)data.Open;
                    features[0, featureIdx++] = (double)data.High;
                    features[0, featureIdx++] = (double)data.Low;
                    features[0, featureIdx++] = (double)data.Close;
                    features[0, featureIdx++] = (double)data.Volume;
                }
                
                // Normalize features
                var normalizedFeatures = _normalizer.Normalize(features);
                
                // Get prediction
                var prediction = await Task.Run(() => _productionModel.Predict(normalizedFeatures));
                double predictedReturn = prediction[0];
                
                // Determine recommendation
                string recommendation;
                if (predictedReturn > 0.01)
                {
                    recommendation = "BUY";
                }
                else if (predictedReturn < -0.01)
                {
                    recommendation = "SHORT";
                }
                else
                {
                    recommendation = "HOLD";
                }
                
                recommendations.Add((symbol, predictedReturn, recommendation));
            }
            catch (Exception ex)
            {
                _logger.Log(LoggingLevel.Warning, $"Error getting prediction for {symbol}: {ex.Message}");
            }
        }
        
        // Sort by absolute predicted return (highest to lowest)
        return recommendations.OrderByDescending(r => Math.Abs(r.PredictedReturn)).ToList();
    }
    
    private static void PrintTradeRecommendations(List<(string Symbol, double PredictedReturn, string Recommendation)> recommendations)
    {
        if (recommendations.Count == 0)
        {
            Console.WriteLine("No trade recommendations available");
            Console.WriteLine();
            return;
        }
        
        Console.WriteLine("TRADE RECOMMENDATIONS:");
        Console.WriteLine(new string('-', 50));
        Console.WriteLine($"{"Symbol",-8}{"Pred. Return",-15}{"Recommendation",-15}");
        Console.WriteLine(new string('-', 50));
        
        foreach (var (symbol, predictedReturn, recommendation) in recommendations)
        {
            string returnClass = predictedReturn >= 0 ? "+" : "";
            
            Console.WriteLine(
                $"{symbol,-8}{returnClass}{predictedReturn:P2,-15}{recommendation,-15}");
        }
        
        Console.WriteLine();
    }
    
    private static async Task ExecuteRecommendedTrades(List<(string Symbol, double PredictedReturn, string Recommendation)> recommendations)
    {
        if (recommendations.Count == 0 || _openPositions.Count >= _tradingConfig.MaxPositions)
        {
            Console.WriteLine("Cannot execute trades: No recommendations or maximum positions reached");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
            return;
        }
        
        var validRecommendations = recommendations
            .Where(r => r.Recommendation != "HOLD")
            .Take(_tradingConfig.MaxPositions - _openPositions.Count)
            .ToList();
            
        if (validRecommendations.Count == 0)
        {
            Console.WriteLine("No actionable trade recommendations");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
            return;
        }
        
        Console.WriteLine("\nExecuting recommended trades:");
        
        foreach (var (symbol, predictedReturn, recommendation) in validRecommendations)
        {
            var latestData = _recentData[symbol].OrderByDescending(d => d.Date).First();
            decimal price = latestData.Close;
            
            // Calculate position size
            decimal positionSize = _availableCash * _tradingConfig.MaxPositionSizePercent;
            decimal commission = positionSize * _tradingConfig.CommissionPerTrade;
            decimal actualInvestment = positionSize - commission;
            decimal quantity = Math.Floor(actualInvestment / price);
            
            if (quantity <= 0)
            {
                Console.WriteLine($"{symbol}: Not enough cash for trade");
                continue;
            }
            
            if (recommendation == "BUY")
            {
                // Create long position
                var position = new Position
                {
                    Symbol = symbol,
                    EntryPrice = price,
                    CurrentPrice = price,
                    Quantity = quantity,
                    EntryDate = DateTime.Now,
                    Type = PositionType.Long,
                    StopLoss = price * (1 - _tradingConfig.StopLossPercent),
                    TakeProfit = price * (1 + _tradingConfig.TakeProfitPercent)
                };
                
                // Update cash and add position
                _availableCash -= (quantity * price) + commission;
                _openPositions.Add(position);
                
                _logger.Log(LoggingLevel.Information, 
                    $"Opened LONG position in {symbol} at {price:C2}, " +
                    $"Quantity: {quantity}, Commission: {commission:C2}");
                    
                Console.WriteLine($"{symbol}: Bought {quantity} shares at {price:C2}");
            }
            else if (recommendation == "SHORT")
            {
                // Create short position
                var position = new Position
                {
                    Symbol = symbol,
                    EntryPrice = price,
                    CurrentPrice = price,
                    Quantity = quantity,
                    EntryDate = DateTime.Now,
                    Type = PositionType.Short,
                    StopLoss = price * (1 + _tradingConfig.StopLossPercent),
                    TakeProfit = price * (1 - _tradingConfig.TakeProfitPercent)
                };
                
                // Update cash and add position (only commission is deducted for short)
                _availableCash -= commission;
                _openPositions.Add(position);
                
                _logger.Log(LoggingLevel.Information, 
                    $"Opened SHORT position in {symbol} at {price:C2}, " +
                    $"Quantity: {quantity}, Commission: {commission:C2}");
                    
                Console.WriteLine($"{symbol}: Shorted {quantity} shares at {price:C2}");
            }
        }
        
        Console.WriteLine("\nPress any key to continue...");
        Console.ReadKey();
        
        await Task.CompletedTask; // For async consistency
    }
    
    private static async Task ClosePosition()
    {
        if (_openPositions.Count == 0)
        {
            Console.WriteLine("No open positions to close");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
            return;
        }
        
        Console.WriteLine("\nSelect position to close:");
        
        for (int i = 0; i < _openPositions.Count; i++)
        {
            var position = _openPositions[i];
            string type = position.Type == PositionType.Long ? "LONG" : "SHORT";
            Console.WriteLine($"{i+1}. {position.Symbol} ({type}) - P/L: {position.ProfitLossPercent:P2}");
        }
        
        Console.Write("\nEnter position number (or 0 to cancel): ");
        var choice = Console.ReadLine()?.Trim();
        
        if (int.TryParse(choice, out int positionIndex) && positionIndex > 0 && positionIndex <= _openPositions.Count)
        {
            var position = _openPositions[positionIndex - 1];
            var latestData = _recentData[position.Symbol].OrderByDescending(d => d.Date).First();
            decimal closePrice = latestData.Close;
            decimal commission = closePrice * position.Quantity * _tradingConfig.CommissionPerTrade;
            
            if (position.Type == PositionType.Long)
            {
                decimal proceeds = position.Quantity * closePrice - commission;
                _availableCash += proceeds;
                
                _logger.Log(LoggingLevel.Information, 
                    $"Closed LONG position in {position.Symbol} at {closePrice:C2}, " +
                    $"Quantity: {position.Quantity}, PL: {position.ProfitLossPercent:P2}");
                    
                Console.WriteLine($"Sold {position.Quantity} shares of {position.Symbol} at {closePrice:C2}");
                Console.WriteLine($"Profit/Loss: {position.ProfitLossPercent:P2}");
            }
            else // Short position
            {
                decimal proceeds = position.Quantity * (position.EntryPrice - closePrice) - commission;
                _availableCash += position.Quantity * position.EntryPrice + proceeds;
                
                _logger.Log(LoggingLevel.Information, 
                    $"Closed SHORT position in {position.Symbol} at {closePrice:C2}, " +
                    $"Quantity: {position.Quantity}, PL: {position.ProfitLossPercent:P2}");
                    
                Console.WriteLine($"Covered {position.Quantity} shares of {position.Symbol} at {closePrice:C2}");
                Console.WriteLine($"Profit/Loss: {position.ProfitLossPercent:P2}");
            }
            
            _openPositions.RemoveAt(positionIndex - 1);
            
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
        
        await Task.CompletedTask; // For async consistency
    }
    
    private static async Task RefreshMarketData()
    {
        Console.WriteLine("Refreshing market data...");
        
        // In a real system, this would fetch the latest prices from a broker API
        // For this example, we'll simulate a price update
        
        Random rand = new Random();
        
        foreach (var symbol in _recentData.Keys)
        {
            var latestData = _recentData[symbol].OrderByDescending(d => d.Date).First();
            var newDate = DateTime.Now;
            
            double priceChange = (rand.NextDouble() * 0.02) - 0.01; // -1% to +1%
            decimal newPrice = latestData.Close * (1 + (decimal)priceChange);
            
            // Create new data point
            var newData = new StockData
            {
                Symbol = symbol,
                Date = newDate,
                Open = latestData.Close,
                Close = newPrice,
                High = Math.Max(latestData.Close, newPrice) * (1 + (decimal)(rand.NextDouble() * 0.005)),
                Low = Math.Min(latestData.Close, newPrice) * (1 - (decimal)(rand.NextDouble() * 0.005)),
                AdjustedClose = newPrice,
                Volume = latestData.Volume + (long)(rand.NextDouble() * 1000000 - 500000)
            };
            
            _recentData[symbol].Add(newData);
            
            // Update positions with new prices
            foreach (var position in _openPositions.Where(p => p.Symbol == symbol))
            {
                position.CurrentPrice = newPrice;
            }
        }
        
        _logger.Log(LoggingLevel.Information, "Market data refreshed");
        
        Console.WriteLine("Market data refreshed. Press any key to continue...");
        Console.ReadKey();
        
        await Task.CompletedTask; // For async consistency
    }
}