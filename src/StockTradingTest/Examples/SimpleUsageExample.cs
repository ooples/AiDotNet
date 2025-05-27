using Microsoft.Extensions.Configuration;
using StockTradingTest.Configuration;
using StockTradingTest.Services;
using StockTradingTest.Models;
using AiDotNet.Interfaces;
using System.Text.Json;

namespace StockTradingTest.Examples
{
    public class SimpleUsageExample
    {
        public static async Task RunExample()
        {
            Console.WriteLine("Starting Simple Stock Trading Test Example");
            
            // Load configuration
            var config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();

            var tradingConfig = new TradingSimulationConfig();
            config.GetSection("TradingSimulation").Bind(tradingConfig);

            var competitionConfig = new ModelCompetitionConfig();
            config.GetSection("ModelCompetition").Bind(competitionConfig);

            var dataConfig = new DataSourceConfig();
            config.GetSection("DataSource").Bind(dataConfig);

            var loggingConfig = new LoggingConfig();
            config.GetSection("Logging").Bind(loggingConfig);

            // Initialize logger
            var logger = new SimulationLogger(loggingConfig);
            logger.Log(StockTradingLogLevel.Information, "Simple Example Started");

            try
            {
                // Create and load sample data
                await PrepareSampleData(dataConfig);
                
                // Create data service
                var dataService = new StockDataService(dataConfig);
                await dataService.LoadDataAsync();
                
                // Create model trainer with just a few models for quick testing
                var modelTrainer = new ModelTrainer(dataService, competitionConfig);
                var competitors = new List<ModelCompetitor>
                {
                    CreateSimpleNeuralNetwork("Simple Neural Network"),
                    CreateSimpleRandomForest("Simple Random Forest")
                };
                
                // Train models on a single symbol
                string testSymbol = dataConfig.DefaultSymbols.First();
                foreach (var competitor in competitors)
                {
                    await modelTrainer.TrainModelAsync(competitor, testSymbol);
                    
                    // Print training metrics
                    Console.WriteLine($"Model: {competitor.Name}");
                    Console.WriteLine($"Training MAE: {competitor.TrainingMetrics.MeanAbsoluteError:F4}");
                    Console.WriteLine($"Training RMSE: {competitor.TrainingMetrics.RootMeanSquaredError:F4}");
                    Console.WriteLine($"Training R²: {competitor.TrainingMetrics.R2Score:F4}");
                    Console.WriteLine($"Training Directional Accuracy: {competitor.TrainingMetrics.DirectionalAccuracy:P2}");
                    Console.WriteLine();
                }
                
                // Create trading service
                var tradingService = new TradingSimulationService(tradingConfig, logger);
                
                // Run a simple simulation for each model
                DateTime startDate = dataConfig.StartDate.AddDays(competitionConfig.LookbackPeriod);
                DateTime endDate = startDate.AddDays(tradingConfig.SimulationDays);
                
                Console.WriteLine($"Running simulation from {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");
                
                List<TournamentResult> results = new List<TournamentResult>();
                foreach (var competitor in competitors)
                {
                    var result = await tradingService.RunSimulation(
                        competitor,
                        dataService,
                        startDate,
                        endDate,
                        1, // round
                        competitionConfig.LookbackPeriod,
                        competitionConfig.PredictionHorizon
                    );
                    
                    results.Add(result);
                }
                
                // Print results
                Console.WriteLine("\nSimulation Results:");
                Console.WriteLine(new string('-', 80));
                Console.WriteLine($"{"Model",-25}{"Profit",-15}{"Win Rate",-15}{"Trades",-10}{"Sharpe",-10}");
                Console.WriteLine(new string('-', 80));
                
                foreach (var result in results.OrderByDescending(r => r.FinalProfitLossPercent))
                {
                    Console.WriteLine(
                        $"{result.ModelName,-25}{result.FinalProfitLossPercent:P2,-15}" +
                        $"{result.WinRate:P2,-15}{result.TotalTrades,-10}{result.Sharpe:F2,-10}");
                }
                
                // Save winner model
                var winner = results.OrderByDescending(r => r.FinalProfitLossPercent).First();
                var winnerModel = competitors.First(c => c.Id == winner.ModelId).Model;
                
                SaveModel(winnerModel, "BestTradingModel.json");
                
                Console.WriteLine($"\nBest model ({winner.ModelName}) saved to BestTradingModel.json");
                Console.WriteLine($"Final balance: {winner.FinalBalance:C2} from {winner.InitialBalance:C2}");
                Console.WriteLine($"Total trades: {winner.TotalTrades}");
                
                logger.Log(StockTradingLogLevel.Information, "Simple Example Completed Successfully");
            }
            catch (Exception ex)
            {
                logger.Log(StockTradingLogLevel.Error, $"Error in simple example: {ex.Message}");
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
        
        private static async Task PrepareSampleData(DataSourceConfig config)
        {
            // Create directory if it doesn't exist
            if (!Directory.Exists(config.StockDataPath))
            {
                Directory.CreateDirectory(config.StockDataPath);
            }
            
            // Here we would normally download real stock data
            // For this example, we'll create synthetic data for testing
            
            foreach (var symbol in config.DefaultSymbols)
            {
                string filePath = Path.Combine(config.StockDataPath, $"{symbol}.csv");
                
                if (!File.Exists(filePath))
                {
                    Console.WriteLine($"Generating synthetic data for {symbol}");
                    await GenerateSyntheticStockData(filePath, symbol, config.StartDate, config.EndDate);
                }
            }
        }
        
        private static async Task GenerateSyntheticStockData(string filePath, string symbol, DateTime startDate, DateTime endDate)
        {
            using var writer = new StreamWriter(filePath);
            
            // Write header
            await writer.WriteLineAsync("Date,Open,High,Low,Close,Adj Close,Volume");
            
            // Generate data with some randomness but a general trend
            Random rand = new Random(symbol.GetHashCode()); // Seed based on symbol for reproducibility
            
            // Initial price based on symbol hash (just for variety)
            double basePrice = 50.0 + (symbol.GetHashCode() % 20) * 10;
            double price = basePrice;
            double volatility = 0.02; // 2% daily volatility
            double drift = 0.0001; // small upward drift
            
            // Add some market regime changes
            List<(DateTime start, DateTime end, double trendMultiplier)> regimes = new List<(DateTime, DateTime, double)>
            {
                (startDate, startDate.AddMonths(6), 0.5),         // Slow growth
                (startDate.AddMonths(6), startDate.AddMonths(9), -2.0),  // Downturn
                (startDate.AddMonths(9), startDate.AddMonths(18), 3.0),  // Strong recovery
                (startDate.AddMonths(18), endDate, 0.2)          // Sideways
            };
            
            for (DateTime date = startDate; date <= endDate; date = date.AddDays(1))
            {
                // Skip weekends
                if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
                    continue;
                
                // Get current regime
                var regime = regimes.FirstOrDefault(r => date >= r.start && date < r.end);
                double currentDrift = drift * regime.trendMultiplier;
                
                // Calculate daily change
                double dailyReturn = rand.NextDouble() * volatility * 2 - volatility + currentDrift;
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
                
                // Write the data
                await writer.WriteLineAsync(
                    $"{date:yyyy-MM-dd},{open:F2},{high:F2},{low:F2},{close:F2},{close:F2},{volume}");
                
                // Update price for next day
                price = close;
            }
        }
        
        private static ModelCompetitor CreateSimpleNeuralNetwork(string name)
        {
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

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.NeuralNetwork
            };
        }
        
        private static ModelCompetitor CreateSimpleRandomForest(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new AiDotNet.Models.Options.RandomForestRegressionOptions
            {
                NumberOfTrees = 50,
                MaxDepth = 10,
                MinSamplesSplit = 2,
                MaxFeatures = 0.7
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.RandomForest
            };
        }
        
        private static void SaveModel(IFullModel<double, Vector<double>, Vector<double>> model, string filePath)
        {
            // In a real implementation, we would serialize the model properly
            // For this example, we just create a placeholder file
            File.WriteAllText(filePath, "Model saved placeholder - this would contain the serialized model data");
        }
    }
}