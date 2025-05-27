using StockTradingTest.Configuration;
using StockTradingTest.Models;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Normalizers;

namespace StockTradingTest.Services
{
    public class TradingSimulationService
    {
        private readonly TradingSimulationConfig _config;
        private readonly SimulationLogger _logger;
        private readonly MeanVarianceNormalizer<double, Vector<double>, Vector<double>> _normalizer = new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>();

        public TradingSimulationService(TradingSimulationConfig config, SimulationLogger logger)
        {
            _config = config;
            _logger = logger;
        }

        public async Task<TournamentResult> RunSimulation(
            ModelCompetitor competitor,
            StockDataService dataService,
            DateTime startDate,
            DateTime endDate,
            int round,
            int lookbackPeriod,
            int predictionHorizon)
        {
            // Initialize portfolio
            var portfolio = new Portfolio(_config.InitialBalance, _config.MaxPositions);
            var tradingDays = dataService.GetTradingDays(startDate, endDate, dataService.Symbols.First());
            var trades = new List<TradeAction>();
            var snapshots = new List<PortfolioSnapshot>();
            
            decimal highestPortfolioValue = portfolio.TotalValue;
            decimal lowestDrawdown = 0;
            
            // Daily returns for performance metrics
            var dailyReturns = new List<double>();
            var previousValue = portfolio.TotalValue;

            _logger.Log(StockTradingLogLevel.Information, 
                $"Starting simulation for model {competitor.Name} (Round {round}), " +
                $"Period: {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");

            foreach (var tradingDay in tradingDays)
            {
                // Check for stop losses and take profits first
                await CheckExistingPositions(portfolio, dataService, tradingDay, trades, competitor.Id, competitor.Name);
                
                // Update portfolio value with current prices
                UpdatePortfolioValues(portfolio, dataService, tradingDay);
                
                // Get predictions and make trading decisions
                if (portfolio.CanOpenPosition)
                {
                    await MakeTradingDecisions(
                        competitor, 
                        dataService, 
                        portfolio, 
                        tradingDay,
                        lookbackPeriod,
                        predictionHorizon,
                        trades
                    );
                }
                
                // Calculate drawdown
                if (portfolio.TotalValue > highestPortfolioValue)
                {
                    highestPortfolioValue = portfolio.TotalValue;
                }
                
                decimal currentDrawdown = (highestPortfolioValue - portfolio.TotalValue) / highestPortfolioValue;
                if (currentDrawdown > lowestDrawdown)
                {
                    lowestDrawdown = currentDrawdown;
                }
                
                // Record daily return
                decimal dailyReturn = portfolio.TotalValue / previousValue - 1.0m;
                dailyReturns.Add((double)dailyReturn);
                previousValue = portfolio.TotalValue;
                
                // Save portfolio snapshot
                snapshots.Add(new PortfolioSnapshot
                {
                    Date = tradingDay,
                    TotalValue = portfolio.TotalValue,
                    Cash = portfolio.Cash,
                    NumPositions = portfolio.Positions.Count,
                    ProfitLoss = portfolio.TotalProfitLoss,
                    ProfitLossPercent = portfolio.TotalProfitLossPercent
                });
                
                _logger.Log(StockTradingLogLevel.Debug, 
                    $"Day: {tradingDay:yyyy-MM-dd}, Portfolio Value: {portfolio.TotalValue:C2}, " +
                    $"Positions: {portfolio.Positions.Count}, Return: {dailyReturn:P2}");
            }
            
            // Close all positions at the end of simulation
            foreach (var position in portfolio.Positions.ToList())
            {
                await ClosePosition(portfolio, position, dataService, tradingDays.Last(), trades, competitor.Id, "End of simulation");
            }
            
            // Calculate performance metrics
            double sharpe = CalculateSharpe(dailyReturns);
            double sortino = CalculateSortino(dailyReturns);
            double calmar = lowestDrawdown == 0 ? 0 : (double)(portfolio.TotalProfitLossPercent / lowestDrawdown);
            
            int winningTrades = trades.Count(t => 
                (t.Type == TradeType.Sell && t.Price > t.Price / (1 + _config.CommissionPerTrade)) || 
                (t.Type == TradeType.ShortCover && t.Price < t.Price * (1 + _config.CommissionPerTrade)));
                
            int losingTrades = trades.Count - winningTrades;
            
            // Create tournament result
            var result = new TournamentResult
            {
                ModelId = competitor.Id,
                ModelName = competitor.Name,
                Round = round,
                InitialBalance = _config.InitialBalance,
                FinalBalance = portfolio.TotalValue,
                TotalTrades = trades.Count,
                WinningTrades = winningTrades,
                LosingTrades = losingTrades,
                MaxDrawdown = lowestDrawdown,
                Sharpe = sharpe,
                Sortino = sortino,
                CalmarRatio = calmar,
                Trades = trades,
                DailySnapshots = snapshots
            };
            
            _logger.Log(StockTradingLogLevel.Information, 
                $"Simulation completed for model {competitor.Name} (Round {round}). " +
                $"Final balance: {portfolio.TotalValue:C2}, PL: {portfolio.TotalProfitLossPercent:P2}, " +
                $"Trades: {trades.Count}, Win rate: {result.WinRate:P2}");
                
            return result;
        }
        
        private async Task CheckExistingPositions(
            Portfolio portfolio, 
            StockDataService dataService, 
            DateTime currentDate,
            List<TradeAction> trades,
            string modelId,
            string modelName)
        {
            foreach (var position in portfolio.Positions.ToList())
            {
                var currentData = dataService.GetStockData(position.Symbol, currentDate);
                if (currentData == null) continue;
                
                position.CurrentPrice = currentData.Close;
                
                // Check stop loss
                if (position.Type == PositionType.Long && currentData.Low <= position.StopLoss)
                {
                    await ClosePosition(
                        portfolio, 
                        position, 
                        dataService, 
                        currentDate, 
                        trades, 
                        modelId, 
                        "Stop loss triggered"
                    );
                    continue;
                }
                else if (position.Type == PositionType.Short && currentData.High >= position.StopLoss)
                {
                    await ClosePosition(
                        portfolio, 
                        position, 
                        dataService, 
                        currentDate, 
                        trades, 
                        modelId, 
                        "Stop loss triggered"
                    );
                    continue;
                }
                
                // Check take profit
                if (position.Type == PositionType.Long && currentData.High >= position.TakeProfit)
                {
                    await ClosePosition(
                        portfolio, 
                        position, 
                        dataService, 
                        currentDate, 
                        trades, 
                        modelId, 
                        "Take profit triggered"
                    );
                    continue;
                }
                else if (position.Type == PositionType.Short && currentData.Low <= position.TakeProfit)
                {
                    await ClosePosition(
                        portfolio, 
                        position, 
                        dataService, 
                        currentDate, 
                        trades, 
                        modelId, 
                        "Take profit triggered"
                    );
                    continue;
                }
            }
        }
        
        private void UpdatePortfolioValues(Portfolio portfolio, StockDataService dataService, DateTime currentDate)
        {
            foreach (var position in portfolio.Positions)
            {
                var currentData = dataService.GetStockData(position.Symbol, currentDate);
                if (currentData != null)
                {
                    position.CurrentPrice = currentData.Close;
                }
            }
        }
        
        private async Task MakeTradingDecisions(
            ModelCompetitor competitor,
            StockDataService dataService,
            Portfolio portfolio,
            DateTime currentDate,
            int lookbackPeriod,
            int predictionHorizon,
            List<TradeAction> trades)
        {
            var predictions = await GetPredictionsForAllSymbols(
                competitor, 
                dataService, 
                currentDate, 
                lookbackPeriod, 
                predictionHorizon
            );
            
            // Sort predictions by predicted return (highest to lowest)
            var sortedPredictions = predictions
                .OrderByDescending(p => p.Value)
                .ToList();
            
            foreach (var prediction in sortedPredictions)
            {
                string symbol = prediction.Key;
                double predictedReturn = prediction.Value;
                
                // Skip if we already have a position in this symbol
                if (portfolio.HasPosition(symbol)) continue;
                
                // Skip if we are at max positions
                if (!portfolio.CanOpenPosition) break;
                
                var stockData = dataService.GetStockData(symbol, currentDate);
                if (stockData == null) continue;
                
                if (predictedReturn > 0.01) // Buy signal
                {
                    await OpenLongPosition(
                        portfolio,
                        symbol,
                        stockData.Close,
                        currentDate,
                        trades,
                        competitor.Id,
                        $"Model predicted +{predictedReturn:P2}"
                    );
                }
                else if (predictedReturn < -0.01) // Short signal
                {
                    await OpenShortPosition(
                        portfolio,
                        symbol,
                        stockData.Close,
                        currentDate,
                        trades,
                        competitor.Id,
                        $"Model predicted {predictedReturn:P2}"
                    );
                }
            }
        }
        
        private async Task<Dictionary<string, double>> GetPredictionsForAllSymbols(
            ModelCompetitor competitor,
            StockDataService dataService,
            DateTime currentDate,
            int lookbackPeriod,
            int predictionHorizon)
        {
            var predictions = new Dictionary<string, double>();
            
            foreach (var symbol in dataService.Symbols)
            {
                try
                {
                    // Get historical data up to current date
                    var (features, _) = dataService.PrepareModelData(
                        symbol,
                        currentDate.AddDays(-lookbackPeriod * 2), // Buffer for dates
                        currentDate,
                        lookbackPeriod,
                        predictionHorizon,
                        false // Don't include targets
                    );
                    
                    // Normalize features
                    var normalizedFeatures = _normalizer.Normalize(features);
                    
                    // Get prediction for the most recent data point only
                    var lastFeatures = new Matrix(1, normalizedFeatures.ColumnCount);
                    for (int i = 0; i < normalizedFeatures.ColumnCount; i++)
                    {
                        lastFeatures[0, i] = normalizedFeatures[normalizedFeatures.Rows - 1, i];
                    }
                    
                    // Make prediction
                    var prediction = await Task.Run(() => competitor.Model.Predict(lastFeatures));
                    predictions[symbol] = prediction[0];
                }
                catch (Exception ex)
                {
                    _logger.Log(StockTradingLogLevel.Warning, $"Failed to get prediction for {symbol}: {ex.Message}");
                    predictions[symbol] = 0; // Neutral on error
                }
            }
            
            return predictions;
        }
        
        private async Task OpenLongPosition(
            Portfolio portfolio,
            string symbol,
            decimal price,
            DateTime date,
            List<TradeAction> trades,
            string modelId,
            string reason)
        {
            decimal positionSize = portfolio.Cash * _config.MaxPositionSizePercent;
            decimal commission = positionSize * _config.CommissionPerTrade;
            decimal actualInvestment = positionSize - commission;
            decimal quantity = Math.Floor(actualInvestment / price);
            
            if (quantity <= 0 || actualInvestment <= 0)
            {
                return;
            }
            
            // Create position
            var position = new Position
            {
                Symbol = symbol,
                EntryPrice = price,
                CurrentPrice = price,
                Quantity = quantity,
                EntryDate = date,
                Type = PositionType.Long,
                StopLoss = price * (1 - _config.StopLossPercent),
                TakeProfit = price * (1 + _config.TakeProfitPercent)
            };
            
            // Update portfolio
            portfolio.Cash -= (quantity * price) + commission;
            portfolio.Positions.Add(position);
            
            // Record trade
            trades.Add(new TradeAction
            {
                Type = TradeType.Buy,
                Symbol = symbol,
                Quantity = quantity,
                Price = price,
                Date = date,
                Commission = commission,
                ModelId = modelId,
                Reason = reason
            });
            
            _logger.Log(StockTradingLogLevel.Information, 
                $"Opened LONG position in {symbol} at {price:C2}, " +
                $"Quantity: {quantity}, Commission: {commission:C2}, Reason: {reason}");
                
            await Task.CompletedTask;
        }
        
        private async Task OpenShortPosition(
            Portfolio portfolio,
            string symbol,
            decimal price,
            DateTime date,
            List<TradeAction> trades,
            string modelId,
            string reason)
        {
            decimal positionSize = portfolio.Cash * _config.MaxPositionSizePercent;
            decimal commission = positionSize * _config.CommissionPerTrade;
            decimal actualInvestment = positionSize - commission;
            decimal quantity = Math.Floor(actualInvestment / price);
            
            if (quantity <= 0 || actualInvestment <= 0)
            {
                return;
            }
            
            // Create position
            var position = new Position
            {
                Symbol = symbol,
                EntryPrice = price,
                CurrentPrice = price,
                Quantity = quantity,
                EntryDate = date,
                Type = PositionType.Short,
                StopLoss = price * (1 + _config.StopLossPercent),
                TakeProfit = price * (1 - _config.TakeProfitPercent)
            };
            
            // Update portfolio
            portfolio.Cash -= commission;
            portfolio.Positions.Add(position);
            
            // Record trade
            trades.Add(new TradeAction
            {
                Type = TradeType.ShortSell,
                Symbol = symbol,
                Quantity = quantity,
                Price = price,
                Date = date,
                Commission = commission,
                ModelId = modelId,
                Reason = reason
            });
            
            _logger.Log(StockTradingLogLevel.Information, 
                $"Opened SHORT position in {symbol} at {price:C2}, " +
                $"Quantity: {quantity}, Commission: {commission:C2}, Reason: {reason}");
                
            await Task.CompletedTask;
        }
        
        private async Task ClosePosition(
            Portfolio portfolio,
            Position position,
            StockDataService dataService,
            DateTime currentDate,
            List<TradeAction> trades,
            string modelId,
            string reason)
        {
            var stockData = dataService.GetStockData(position.Symbol, currentDate);
            if (stockData == null) return;
            
            decimal closePrice = stockData.Close;
            decimal commission = closePrice * position.Quantity * _config.CommissionPerTrade;
            
            if (position.Type == PositionType.Long)
            {
                decimal proceeds = position.Quantity * closePrice - commission;
                portfolio.Cash += proceeds;
                
                trades.Add(new TradeAction
                {
                    Type = TradeType.Sell,
                    Symbol = position.Symbol,
                    Quantity = position.Quantity,
                    Price = closePrice,
                    Date = currentDate,
                    Commission = commission,
                    ModelId = modelId,
                    Reason = reason
                });
                
                _logger.Log(StockTradingLogLevel.Information, 
                    $"Closed LONG position in {position.Symbol} at {closePrice:C2}, " +
                    $"Quantity: {position.Quantity}, PL: {position.ProfitLossPercent:P2}, Reason: {reason}");
            }
            else // Short position
            {
                decimal proceeds = position.Quantity * (position.EntryPrice - closePrice) - commission;
                portfolio.Cash += position.Quantity * position.EntryPrice + proceeds;
                
                trades.Add(new TradeAction
                {
                    Type = TradeType.ShortCover,
                    Symbol = position.Symbol,
                    Quantity = position.Quantity,
                    Price = closePrice,
                    Date = currentDate,
                    Commission = commission,
                    ModelId = modelId,
                    Reason = reason
                });
                
                _logger.Log(StockTradingLogLevel.Information, 
                    $"Closed SHORT position in {position.Symbol} at {closePrice:C2}, " +
                    $"Quantity: {position.Quantity}, PL: {position.ProfitLossPercent:P2}, Reason: {reason}");
            }
            
            // Remove position from portfolio
            portfolio.Positions.Remove(position);
            
            await Task.CompletedTask;
        }
        
        private double CalculateSharpe(List<double> returns, double riskFreeRate = 0.0)
        {
            if (returns.Count <= 1) return 0;
            
            double meanReturn = returns.Average();
            double stdDev = Math.Sqrt(returns.Select(r => Math.Pow(r - meanReturn, 2)).Sum() / (returns.Count - 1));
            
            if (stdDev == 0) return 0;
            
            // Annualize (assuming daily returns)
            double annualizedSharpe = (meanReturn - riskFreeRate) / stdDev * Math.Sqrt(252);
            return annualizedSharpe;
        }
        
        private double CalculateSortino(List<double> returns, double riskFreeRate = 0.0)
        {
            if (returns.Count <= 1) return 0;
            
            double meanReturn = returns.Average();
            var downReturns = returns.Where(r => r < 0).Select(r => r * r).ToList();
            
            if (downReturns.Count == 0) return double.MaxValue;
            
            double downDev = Math.Sqrt(downReturns.Sum() / downReturns.Count);
            
            if (downDev == 0) return 0;
            
            // Annualize (assuming daily returns)
            double annualizedSortino = (meanReturn - riskFreeRate) / downDev * Math.Sqrt(252);
            return annualizedSortino;
        }
    }
}