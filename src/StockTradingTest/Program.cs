using Microsoft.Extensions.Configuration;
using StockTradingTest.Configuration;
using StockTradingTest.Services;
using StockTradingTest.Models;
using StockTradingTest.Examples;

namespace StockTradingTest
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== Stock Trading Test System ===");
            Console.WriteLine("1. Run full tournament");
            Console.WriteLine("2. Run simple example");
            Console.WriteLine("3. Run production example");
            Console.Write("Enter choice (1-3): ");
            
            string choice = Console.ReadLine()?.Trim() ?? "1";
            
            switch (choice)
            {
                case "2":
                    await SimpleUsageExample.RunExample();
                    break;
                case "3":
                    await ProductionExample.RunExample();
                    break;
                case "1":
                default:
                    await RunFullTournament();
                    break;
            }
        }
        
        static async Task RunFullTournament()
        {
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
            logger.Log(StockTradingLogLevel.Information, "Stock Trading Simulation Started");

            try
            {
                // Create data directory if needed
                if (!Directory.Exists(dataConfig.StockDataPath))
                {
                    Directory.CreateDirectory(dataConfig.StockDataPath);
                }
                
                // Create data service
                var dataService = new StockDataService(dataConfig);
                await dataService.LoadDataAsync();
                logger.Log(StockTradingLogLevel.Information, $"Loaded data for {dataService.Symbols.Count} symbols");

                // Create model trainer
                var modelTrainer = new ModelTrainer(dataService, competitionConfig);

                // Create trading service
                var tradingService = new TradingSimulationService(tradingConfig, logger);

                // Create tournament
                var tournament = new ModelTournament(
                    modelTrainer,
                    tradingService,
                    competitionConfig,
                    logger
                );

                // Run tournament
                var results = await tournament.RunTournamentAsync();

                // Display results
                var reporter = new TournamentReporter(results, logger);
                reporter.PrintSummary();
                reporter.GenerateDetailedReport("TournamentResults.html");

                logger.Log(StockTradingLogLevel.Information, "Stock Trading Simulation Completed Successfully");
                
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
            }
            catch (Exception ex)
            {
                logger.Log(StockTradingLogLevel.Error, $"Error in simulation: {ex.Message}");
                logger.Log(StockTradingLogLevel.Debug, ex.StackTrace ?? "No stack trace available");
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
            }
        }
    }
}