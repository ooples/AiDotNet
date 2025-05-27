using StockTradingTest.Configuration;
using StockTradingTest.Models;

namespace StockTradingTest.Services
{
    public class ModelTournament
    {
        private readonly ModelTrainer _modelTrainer;
        private readonly TradingSimulationService _tradingService;
        private readonly ModelCompetitionConfig _config;
        private readonly SimulationLogger _logger;
        private readonly StockDataService _dataService;
        private List<ModelCompetitor> _competitors = new List<ModelCompetitor>();
        private readonly List<TournamentResult> _allResults = new List<TournamentResult>();

        public ModelTournament(
            ModelTrainer modelTrainer,
            TradingSimulationService tradingService,
            ModelCompetitionConfig config,
            SimulationLogger logger)
        {
            _modelTrainer = modelTrainer;
            _tradingService = tradingService;
            _config = config;
            _logger = logger;
        }

        public async Task<List<TournamentResult>> RunTournamentAsync()
        {
            _logger.Log(StockTradingLogLevel.Information, $"Starting model tournament with {_config.NumberOfRounds} rounds");
            
            // Create competitors
            _competitors = _modelTrainer.CreateCompetitors();
            _logger.Log(StockTradingLogLevel.Information, $"Created {_competitors.Count} model competitors");
            
            // Run rounds
            for (int round = 1; round <= _config.NumberOfRounds; round++)
            {
                _logger.Log(StockTradingLogLevel.Information, $"Starting tournament round {round}");
                await RunRoundAsync(round);
                
                // Eliminate poorest performers if not the last round
                if (round < _config.NumberOfRounds)
                {
                    EliminateCompetitors();
                }
            }
            
            // Sort final results by performance
            var finalResults = _allResults
                .Where(r => r.Round == _config.NumberOfRounds)
                .OrderByDescending(r => r.FinalProfitLossPercent)
                .ToList();
                
            _logger.Log(StockTradingLogLevel.Information, "Tournament completed. Final rankings:");
            for (int i = 0; i < finalResults.Count; i++)
            {
                _logger.Log(StockTradingLogLevel.Information, 
                    $"{i+1}. {finalResults[i].ModelName} - {finalResults[i].FinalProfitLossPercent:P2}, " +
                    $"Sharpe: {finalResults[i].Sharpe:F2}, Win Rate: {finalResults[i].WinRate:P2}");
            }
            
            return _allResults;
        }
        
        private async Task RunRoundAsync(int round)
        {
            // Calculate dates for this round
            int totalDays = (int)(_config.EndDate - _config.StartDate).TotalDays;
            int daysPerRound = totalDays / _config.NumberOfRounds;
            
            DateTime roundStartDate = _config.StartDate.AddDays(daysPerRound * (round - 1));
            DateTime roundEndDate = round == _config.NumberOfRounds 
                ? _config.EndDate 
                : _config.StartDate.AddDays(daysPerRound * round - 1);
                
            _logger.Log(StockTradingLogLevel.Information, 
                $"Round {round} period: {roundStartDate:yyyy-MM-dd} to {roundEndDate:yyyy-MM-dd}");
                
            // Train models for each competitor
            var activeCompetitors = _competitors.Where(c => !c.IsEliminated).ToList();
            _logger.Log(StockTradingLogLevel.Information, $"Training {activeCompetitors.Count} active models");
            
            List<Task<ModelCompetitor>> trainingTasks = new List<Task<ModelCompetitor>>();
            foreach (var competitor in activeCompetitors)
            {
                // Train on each symbol
                foreach (var symbol in _dataService.Symbols)
                {
                    trainingTasks.Add(_modelTrainer.TrainModelAsync(competitor, symbol));
                }
            }
            
            await Task.WhenAll(trainingTasks);
            
            // Run simulations for each competitor
            _logger.Log(StockTradingLogLevel.Information, $"Running trading simulations for round {round}");
            
            List<Task<TournamentResult>> simulationTasks = new List<Task<TournamentResult>>();
            foreach (var competitor in activeCompetitors)
            {
                simulationTasks.Add(_tradingService.RunSimulation(
                    competitor,
                    _dataService,
                    roundStartDate,
                    roundEndDate,
                    round,
                    _config.LookbackPeriod,
                    _config.PredictionHorizon
                ));
            }
            
            var roundResults = await Task.WhenAll(simulationTasks);
            
            // Store results and add to competitors
            foreach (var result in roundResults)
            {
                _allResults.Add(result);
                
                var competitor = activeCompetitors.First(c => c.Id == result.ModelId);
                competitor.TournamentResults.Add(result);
            }
            
            // Log round summary
            var sortedRoundResults = roundResults
                .OrderByDescending(r => r.FinalProfitLossPercent)
                .ToList();
                
            _logger.Log(StockTradingLogLevel.Information, $"Round {round} completed. Results:");
            for (int i = 0; i < sortedRoundResults.Count; i++)
            {
                _logger.Log(StockTradingLogLevel.Information, 
                    $"{i+1}. {sortedRoundResults[i].ModelName} - {sortedRoundResults[i].FinalProfitLossPercent:P2}, " +
                    $"Trades: {sortedRoundResults[i].TotalTrades}, Win Rate: {sortedRoundResults[i].WinRate:P2}");
            }
        }
        
        private void EliminateCompetitors()
        {
            var activeCompetitors = _competitors.Where(c => !c.IsEliminated).ToList();
            int elimCount = (int)Math.Floor(activeCompetitors.Count * _config.EliminationPercentage);
            
            if (elimCount <= 0) return; // No elimination if too few competitors
            
            // Sort by average profit
            var sortedCompetitors = activeCompetitors
                .OrderBy(c => c.AverageProfitPercent)
                .ToList();
                
            // Eliminate bottom performers
            for (int i = 0; i < elimCount; i++)
            {
                sortedCompetitors[i].IsEliminated = true;
                _logger.Log(StockTradingLogLevel.Information, 
                    $"Eliminated model {sortedCompetitors[i].Name} with average profit {sortedCompetitors[i].AverageProfitPercent:P2}");
            }
            
            _logger.Log(StockTradingLogLevel.Information, 
                $"Eliminated {elimCount} competitors. {activeCompetitors.Count - elimCount} competitors remain.");
        }
    }
}