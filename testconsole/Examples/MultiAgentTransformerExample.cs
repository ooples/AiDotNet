using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Models;
using AiDotNet.ReinforcementLearning.Models.Options;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating how to use the Multi-Agent Transformer model for financial market modeling and trading.
    /// </summary>
    /// <remarks>
    /// This example shows how to set up, train, and evaluate a Multi-Agent Transformer model for stock market prediction and trading.
    /// The model treats the market as a system of interacting agents (different types of traders) and uses transformer
    /// architecture to capture complex market dynamics and agent interactions.
    /// </remarks>
    public class MultiAgentTransformerExample
    {
        /// <summary>
        /// Runs the example.
        /// </summary>
        public static void Run()
        {
            Console.WriteLine("Running Multi-Agent Transformer for Financial Markets Example");
            Console.WriteLine("=============================================================");
            
            // Create a simulated market environment
            var environment = CreateStockMarketEnvironment();
            
            // Create model options
            var options = CreateModelOptions();
            
            // Create and initialize the model
            var model = new MultiAgentTransformerModel<double>(options);
            
            // Train the model
            TrainModel(model, environment, 1000);
            
            // Evaluate the model
            EvaluateModel(model, environment, 100);
            
            // Demonstrate advanced market analysis features
            DemonstrateMarketAnalysis(model, environment);
            
            Console.WriteLine("\nMulti-Agent Transformer Example completed successfully!");
        }
        
        /// <summary>
        /// Creates configuration options for the Multi-Agent Transformer model.
        /// </summary>
        private static MultiAgentTransformerOptions CreateModelOptions()
        {
            return new MultiAgentTransformerOptions
            {
                NumAgents = 4,                              // Model 4 types of market participants
                HiddenDimension = 128,                      // Size of hidden layers
                NumHeads = 8,                               // Number of attention heads
                NumLayers = 4,                              // Number of transformer layers
                SequenceLength = 50,                        // Length of market history to consider
                TransformerLearningRate = 0.0003,           // Learning rate for transformer
                Gamma = 0.99,                               // Discount factor for future rewards
                EntropyCoefficient = 0.01,                  // Encourages exploration
                UseCentralizedTraining = true,              // Share information during training
                RiskAversionParameter = 0.5,                // Moderate risk aversion
                PositionalEncodingType = PositionalEncodingType.Sinusoidal, // Time encoding method
                UseCausalMask = true,                       // Prevent looking into the future
                ModelMarketImpact = true                    // Consider how trades affect the market
            };
        }
        
        /// <summary>
        /// Creates a simulated stock market environment for testing the model.
        /// </summary>
        private static SimulatedStockMarket CreateStockMarketEnvironment()
        {
            Console.WriteLine("Creating simulated stock market environment...");
            
            // Create the environment with some initial parameters
            var environment = new SimulatedStockMarket(
                numStocks: 5,                  // Model 5 different stocks
                initialPrices: new[] { 100.0, 150.0, 200.0, 75.0, 300.0 },
                volatility: 0.02,              // 2% daily volatility
                correlationStrength: 0.3,      // Moderate correlation between stocks
                tradingFee: 0.001,             // 0.1% trading fee
                initialBalance: 100000.0,      // Start with $100,000
                episodeLength: 252,            // 1 year of trading days
                includesTechnicalIndicators: true);  // Include indicators like RSI, MACD, etc.
            
            Console.WriteLine("Environment created with 5 stocks and 1-year episodes.");
            
            return environment;
        }
        
        /// <summary>
        /// Trains the model on the simulated market environment.
        /// </summary>
        private static void TrainModel(MultiAgentTransformerModel<double> model, SimulatedStockMarket environment, int episodes)
        {
            Console.WriteLine($"\nTraining model for {episodes} episodes...");
            
            double totalReward = 0;
            double bestEpisodeReward = double.MinValue;
            
            for (int episode = 0; episode < episodes; episode++)
            {
                // Reset the environment for a new episode
                var state = environment.Reset();
                double episodeReward = 0;
                bool done = false;
                
                while (!done)
                {
                    // Select action using the model
                    var action = model.SelectAction(state, isTraining: true);
                    
                    // Take action in the environment
                    var (nextState, reward, isDone) = environment.Step(action);
                    
                    // Update the model
                    model.Update(state, action, reward, nextState, isDone);
                    
                    // Move to the next state
                    state = nextState;
                    episodeReward += reward;
                    done = isDone;
                }
                
                totalReward += episodeReward;
                bestEpisodeReward = Math.Max(bestEpisodeReward, episodeReward);
                
                // Print progress
                if ((episode + 1) % 100 == 0 || episode == 0)
                {
                    double avgReward = totalReward / (episode + 1);
                    Console.WriteLine($"Episode {episode + 1}/{episodes} - Avg Reward: {avgReward:F2} - Best Reward: {bestEpisodeReward:F2}");
                }
            }
            
            Console.WriteLine("Training completed!");
            Console.WriteLine($"Final Average Reward: {totalReward / episodes:F2}");
            Console.WriteLine($"Best Episode Reward: {bestEpisodeReward:F2}");
        }
        
        /// <summary>
        /// Evaluates the trained model on the simulated market environment.
        /// </summary>
        private static void EvaluateModel(MultiAgentTransformerModel<double> model, SimulatedStockMarket environment, int episodes)
        {
            Console.WriteLine($"\nEvaluating model for {episodes} episodes...");
            
            double totalReward = 0;
            double totalReturnPercent = 0;
            int profitableEpisodes = 0;
            
            for (int episode = 0; episode < episodes; episode++)
            {
                // Reset the environment for a new episode
                var state = environment.Reset();
                double initialBalance = environment.GetBalance();
                double episodeReward = 0;
                bool done = false;
                
                while (!done)
                {
                    // Select action using the model (no exploration during evaluation)
                    var action = model.SelectAction(state, isTraining: false);
                    
                    // Take action in the environment
                    var (nextState, reward, isDone) = environment.Step(action);
                    
                    // Move to the next state
                    state = nextState;
                    episodeReward += reward;
                    done = isDone;
                }
                
                double finalBalance = environment.GetBalance();
                double returnPercent = (finalBalance - initialBalance) / initialBalance * 100;
                
                totalReward += episodeReward;
                totalReturnPercent += returnPercent;
                
                if (finalBalance > initialBalance)
                {
                    profitableEpisodes++;
                }
                
                // Print progress
                if ((episode + 1) % 10 == 0 || episode == 0)
                {
                    Console.WriteLine($"Evaluation Episode {episode + 1}/{episodes} - Return: {returnPercent:F2}% - Final Balance: ${finalBalance:F2}");
                }
            }
            
            // Calculate statistics
            double avgReward = totalReward / episodes;
            double avgReturn = totalReturnPercent / episodes;
            double profitability = (double)profitableEpisodes / episodes * 100;
            
            Console.WriteLine("\nEvaluation Results:");
            Console.WriteLine($"Average Reward: {avgReward:F2}");
            Console.WriteLine($"Average Return: {avgReturn:F2}%");
            Console.WriteLine($"Profitable Episodes: {profitableEpisodes}/{episodes} ({profitability:F2}%)");
        }
        
        /// <summary>
        /// Demonstrates the advanced market analysis features of the Multi-Agent Transformer model.
        /// </summary>
        private static void DemonstrateMarketAnalysis(MultiAgentTransformerModel<double> model, SimulatedStockMarket environment)
        {
            Console.WriteLine("\nDemonstrating Advanced Market Analysis Features");
            Console.WriteLine("-----------------------------------------------");
            
            // Get a sample market state
            var state = environment.Reset();
            
            // 1. Market Regime Detection
            string regime = model.DetectMarketRegime(state);
            Console.WriteLine($"\nDetected Market Regime: {regime}");
            
            // 2. Action Risk Profile Analysis
            Console.WriteLine("\nAction Risk Profile Analysis:");
            var riskProfile = model.AnalyzeActionRiskProfile(state);
            foreach (var entry in riskProfile.OrderByDescending(e => e.Value))
            {
                Console.WriteLine($"  {entry.Key}: Expected Risk-Adjusted Return = {entry.Value:F4}");
            }
            
            // 3. Agent Interaction Analysis
            Console.WriteLine("\nAgent Interaction Analysis:");
            var attentionWeights = model.GetAgentInteractionAttention(state);
            PrintAttentionHeatmap(attentionWeights);
            
            // 4. Future State Prediction
            Console.WriteLine("\nFuture Market State Prediction (5 days ahead):");
            var futureStates = model.PredictFutureStates(state, 5);
            
            for (int day = 0; day < futureStates.Count; day++)
            {
                var predictedPrices = ExtractPredictedPrices(futureStates[day], environment.GetStockCount());
                Console.WriteLine($"  Day {day}: Predicted Prices = [{string.Join(", ", predictedPrices.Select(p => $"${p:F2}"))}]");
            }
            
            // 5. Multi-Agent Action Analysis
            Console.WriteLine("\nMulti-Agent Action Analysis:");
            var allAgentActions = model.SelectActionsForAllAgents(state, false);
            
            string[] agentTypes = { "Retail Trader", "Institutional Investor", "Market Maker", "High-Frequency Trader" };
            for (int i = 0; i < allAgentActions.Count; i++)
            {
                Console.WriteLine($"  {agentTypes[i]}: {FormatAction(allAgentActions[i])}");
            }
        }
        
        /// <summary>
        /// Formats an action vector for display.
        /// </summary>
        private static string FormatAction(Vector<double> action)
        {
            // Assuming action[0] is position size (-1 to 1 scale)
            double positionSize = action[0];
            
            if (positionSize > 0.7)
                return $"Strong Buy (Position Size: {positionSize:F2})";
            else if (positionSize > 0.3)
                return $"Moderate Buy (Position Size: {positionSize:F2})";
            else if (positionSize > -0.3)
                return $"Hold (Position Size: {positionSize:F2})";
            else if (positionSize > -0.7)
                return $"Moderate Sell (Position Size: {positionSize:F2})";
            else
                return $"Strong Sell (Position Size: {positionSize:F2})";
        }
        
        /// <summary>
        /// Extracts predicted stock prices from a state tensor.
        /// </summary>
        private static List<double> ExtractPredictedPrices(Tensor<double> state, int numStocks)
        {
            // Assuming the first numStocks elements in the state tensor are the prices
            var prices = new List<double>();
            for (int i = 0; i < numStocks; i++)
            {
                prices.Add(state[0, i]);
            }
            return prices;
        }
        
        /// <summary>
        /// Prints a heatmap visualization of attention weights between agents.
        /// </summary>
        private static void PrintAttentionHeatmap(Tensor<double> attentionWeights)
        {
            string[] agentTypes = { "Retail", "Instit.", "Market", "HFT" };
            
            // Print header
            Console.Write("          ");
            foreach (var agent in agentTypes)
            {
                Console.Write($"{agent,-10}");
            }
            Console.WriteLine();
            
            // Print rows
            for (int i = 0; i < agentTypes.Length; i++)
            {
                Console.Write($"{agentTypes[i],-10}");
                
                for (int j = 0; j < agentTypes.Length; j++)
                {
                    double weight = attentionWeights[0, i, j];
                    
                    // Convert weight to intensity character
                    char intensityChar = GetIntensityChar(weight);
                    
                    Console.Write($"{intensityChar,-10}");
                }
                Console.WriteLine();
            }
            
            // Print legend
            Console.WriteLine("\nInfluence Scale: None ' ' < . < o < O < @ (Strongest)");
        }
        
        /// <summary>
        /// Converts an attention weight to a character representing its intensity.
        /// </summary>
        private static char GetIntensityChar(double weight)
        {
            if (weight < 0.1) return ' ';
            if (weight < 0.3) return '.';
            if (weight < 0.5) return 'o';
            if (weight < 0.7) return 'O';
            return '@';
        }
        
        /// <summary>
        /// A simulated stock market environment for testing reinforcement learning models.
        /// </summary>
        private class SimulatedStockMarket
        {
            private readonly int _numStocks;
            private readonly double[] _initialPrices;
            private readonly double _volatility;
            private readonly double _correlationStrength;
            private readonly double _tradingFee;
            private readonly int _episodeLength;
            private readonly bool _includesTechnicalIndicators;
            private readonly Random _random = default!;
            
            private double[] _currentPrices;
            private double _balance;
            private int _currentDay;
            private double[] _positions;
            
            /// <summary>
            /// Creates a new instance of the SimulatedStockMarket class.
            /// </summary>
            public SimulatedStockMarket(
                int numStocks,
                double[] initialPrices,
                double volatility,
                double correlationStrength,
                double tradingFee,
                double initialBalance,
                int episodeLength,
                bool includesTechnicalIndicators)
            {
                _numStocks = numStocks;
                _initialPrices = initialPrices;
                _volatility = volatility;
                _correlationStrength = correlationStrength;
                _tradingFee = tradingFee;
                _balance = initialBalance;
                _episodeLength = episodeLength;
                _includesTechnicalIndicators = includesTechnicalIndicators;
                _random = new Random(42); // Fixed seed for reproducibility
                
                _currentPrices = new double[_numStocks];
                _positions = new double[_numStocks];
                Reset();
            }
            
            /// <summary>
            /// Resets the environment to start a new episode.
            /// </summary>
            public Tensor<double> Reset()
            {
                // Reset prices to initial values
                Array.Copy(_initialPrices, _currentPrices, _numStocks);
                
                // Reset positions and balance
                Array.Clear(_positions, 0, _positions.Length);
                _balance = _initialPrices.Sum() * 100; // Start with enough money to buy 100 of each stock
                _currentDay = 0;
                
                // Return initial state
                return GetState();
            }
            
            /// <summary>
            /// Takes a step in the environment based on the given action.
            /// </summary>
            public (Tensor<double>, double, bool) Step(Vector<double> action)
            {
                // Execute trading action
                double reward = ExecuteAction(action);
                
                // Update market prices
                UpdatePrices();
                
                // Increment day counter
                _currentDay++;
                
                // Check if episode is done
                bool isDone = _currentDay >= _episodeLength || _balance <= 0;
                
                // Return new state, reward, and done flag
                return (GetState(), reward, isDone);
            }
            
            /// <summary>
            /// Gets the number of stocks in the environment.
            /// </summary>
            public int GetStockCount() => _numStocks;
            
            /// <summary>
            /// Gets the current balance.
            /// </summary>
            public double GetBalance() => _balance;
            
            /// <summary>
            /// Gets the current state representation.
            /// </summary>
            private Tensor<double> GetState()
            {
                // For simplicity, we'll represent the state as:
                // - Current prices (numStocks elements)
                // - Current positions (numStocks elements)
                // - Price changes over the last 5 days (5 * numStocks elements) - simulated
                // - Technical indicators (if enabled) - simulated
                
                int baseFeatureCount = _numStocks * 2; // Prices and positions
                int priceHistoryFeatures = _numStocks * 5; // 5 days of price history
                int technicalIndicatorFeatures = _includesTechnicalIndicators ? _numStocks * 5 : 0; // 5 indicators per stock
                
                int stateDimension = baseFeatureCount + priceHistoryFeatures + technicalIndicatorFeatures;
                var state = new Tensor<double>(new[] { 1, stateDimension });
                
                int featureIdx = 0;
                
                // Add current prices
                for (int i = 0; i < _numStocks; i++)
                {
                    state[0, featureIdx++] = _currentPrices[i];
                }
                
                // Add current positions
                for (int i = 0; i < _numStocks; i++)
                {
                    state[0, featureIdx++] = _positions[i];
                }
                
                // Add price history (simulated)
                for (int day = 0; day < 5; day++)
                {
                    for (int i = 0; i < _numStocks; i++)
                    {
                        // Simulate past prices with some random noise
                        double historicalPrice = _currentPrices[i] * (1 - 0.01 * (day + 1) * (1 + 0.2 * (_random.NextDouble() - 0.5)));
                        state[0, featureIdx++] = historicalPrice;
                    }
                }
                
                // Add technical indicators (if enabled)
                if (_includesTechnicalIndicators)
                {
                    for (int i = 0; i < _numStocks; i++)
                    {
                        // Simulated RSI (0-100)
                        double rsi = 50 + 20 * (_random.NextDouble() - 0.5);
                        state[0, featureIdx++] = rsi / 100.0; // Normalize to 0-1
                        
                        // Simulated MACD
                        double macd = 0.2 * (_random.NextDouble() - 0.5);
                        state[0, featureIdx++] = macd;
                        
                        // Simulated Bollinger Band width
                        double bbWidth = 0.05 * (1 + 0.5 * _random.NextDouble());
                        state[0, featureIdx++] = bbWidth;
                        
                        // Simulated Volume (relative to average)
                        double volume = 1.0 + 0.5 * (_random.NextDouble() - 0.5);
                        state[0, featureIdx++] = volume;
                        
                        // Simulated ATR (Average True Range)
                        double atr = _currentPrices[i] * 0.02 * (0.5 + _random.NextDouble());
                        state[0, featureIdx++] = atr / _currentPrices[i]; // Normalize by price
                    }
                }
                
                return state;
            }
            
            /// <summary>
            /// Executes the trading action and calculates the reward.
            /// </summary>
            private double ExecuteAction(Vector<double> action)
            {
                // For simplicity, we'll assume:
                // - action[0] is the target position for the first stock (-1 to 1 scale)
                // - action[1] is the target position for the second stock (if available)
                // ... and so on.
                
                double initialPortfolioValue = CalculatePortfolioValue();
                
                // Calculate target positions
                double[] targetPositions = new double[_numStocks];
                for (int i = 0; i < _numStocks && i < action.Length; i++)
                {
                    // Convert action (-1 to 1) to a position size
                    // -1 = full short, 0 = no position, 1 = full long
                    targetPositions[i] = action[i] * 100; // Scale to number of shares
                }
                
                // Execute trades to reach target positions
                for (int i = 0; i < _numStocks; i++)
                {
                    if (_positions[i] != targetPositions[i])
                    {
                        // Calculate the number of shares to buy or sell
                        double sharesToTrade = targetPositions[i] - _positions[i];
                        
                        // Calculate the cost/proceeds of the trade
                        double tradeCost = sharesToTrade * _currentPrices[i];
                        
                        // Add trading fee
                        double fee = Math.Abs(tradeCost) * _tradingFee;
                        tradeCost += fee;
                        
                        // Check if we have enough balance for a buy
                        if (tradeCost > 0 && tradeCost > _balance)
                        {
                            // Not enough money, adjust the trade size
                            sharesToTrade = (_balance / _currentPrices[i]) / (1 + _tradingFee);
                            tradeCost = sharesToTrade * _currentPrices[i] * (1 + _tradingFee);
                        }
                        
                        // Update position and balance
                        _positions[i] += sharesToTrade;
                        _balance -= tradeCost;
                    }
                }
                
                // Calculate reward as the change in portfolio value
                double finalPortfolioValue = CalculatePortfolioValue();
                return finalPortfolioValue - initialPortfolioValue;
            }
            
            /// <summary>
            /// Updates stock prices for the next day.
            /// </summary>
            private void UpdatePrices()
            {
                // Generate a market-wide return (common factor)
                double marketReturn = _volatility * 2 * (_random.NextDouble() - 0.5);
                
                // Update each stock price
                for (int i = 0; i < _numStocks; i++)
                {
                    // Generate stock-specific return
                    double specificReturn = _volatility * 2 * (_random.NextDouble() - 0.5);
                    
                    // Combine market and specific returns based on correlation strength
                    double totalReturn = (_correlationStrength * marketReturn) + ((1 - _correlationStrength) * specificReturn);
                    
                    // Update price with the return
                    _currentPrices[i] *= (1 + totalReturn);
                    
                    // Ensure price doesn't go below a minimum value
                    _currentPrices[i] = Math.Max(_currentPrices[i], 1.0);
                }
            }
            
            /// <summary>
            /// Calculates the current portfolio value.
            /// </summary>
            private double CalculatePortfolioValue()
            {
                double totalValue = _balance;
                
                for (int i = 0; i < _numStocks; i++)
                {
                    totalValue += _positions[i] * _currentPrices[i];
                }
                
                return totalValue;
            }
        }
    }
}