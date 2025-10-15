using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Models;
using AiDotNet.ReinforcementLearning.Models.Options;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating how to use the Hierarchical Risk-Aware RL (HRARL) model for financial market trading.
    /// </summary>
    /// <remarks>
    /// This example shows how to set up, train, and evaluate an HRARL model for stock market prediction and trading.
    /// The model uses hierarchical decision-making with explicit risk consideration, making it particularly
    /// suitable for financial applications where managing risk is as important as maximizing returns.
    /// </remarks>
    public class HRARLExample
    {
        /// <summary>
        /// Runs the example.
        /// </summary>
        public static void Run()
        {
            Console.WriteLine("Running Hierarchical Risk-Aware RL (HRARL) Example");
            Console.WriteLine("==================================================");
            
            // Create a simulated market environment
            var environment = CreateStockMarketEnvironment();
            
            // Create model options
            var options = CreateModelOptions();
            
            // Create and initialize the model
            var model = new HRARLModel<double>(options);
            
            // Train the model
            TrainModel(model, environment, 1000);
            
            // Evaluate the model with different risk aversion settings
            Console.WriteLine("\nEvaluating model with different risk profiles:");
            EvaluateModelWithDifferentRiskSettings(model, environment);
            
            // Demonstrate risk analysis features
            DemonstrateRiskAnalysis(model, environment);
            
            // Perform strategy simulation
            DemonstrateStrategySimulation(model, environment);
            
            Console.WriteLine("\nHierarchical Risk-Aware RL Example completed successfully!");
        }
        
        /// <summary>
        /// Creates configuration options for the HRARL model.
        /// </summary>
        private static HRARLOptions CreateModelOptions()
        {
            return new HRARLOptions
            {
                NumHierarchicalLevels = 2,                // Two-level hierarchy (strategic and tactical)
                HighLevelHiddenDimension = 256,           // Size of high-level network layers
                LowLevelHiddenDimension = 128,            // Size of low-level network layers
                RiskAversionParameter = 0.5,              // Moderate risk aversion
                UseAdaptiveRiskAversion = true,           // Adjust risk aversion based on market conditions
                HighLevelTimeHorizon = 30,                // High-level makes decisions every 30 steps
                LowLevelTimeHorizon = 5,                  // Low-level makes decisions every 5 steps
                HighLevelGamma = 0.99,                    // High discount factor for high-level (long-term focus)
                LowLevelGamma = 0.97,                     // Lower discount factor for low-level
                HighLevelLearningRate = 0.0001,           // Slower learning for high-level
                LowLevelLearningRate = 0.0003,            // Faster learning for low-level
                HighLevelEntropyCoef = 0.01,              // Exploration parameter for high-level
                LowLevelEntropyCoef = 0.02,               // Exploration parameter for low-level
                RiskMetricType = 2,                       // Use CVaR as risk metric (most conservative)
                ConfidenceLevel = 0.05,                   // 5% confidence level (95% CVaR)
                UseRecurrentHighLevelPolicy = true,       // Use LSTM for high-level policy
                UseIntrinsicRewards = true,               // Encourage exploration
                UseTargetNetwork = true,                  // Use target networks for stable learning
                UseHindsightExperienceReplay = true       // Learn from alternative outcomes
            };
        }
        
        /// <summary>
        /// Creates a simulated stock market environment for testing the model.
        /// </summary>
        private static FinancialMarketEnvironment CreateStockMarketEnvironment()
        {
            Console.WriteLine("Creating simulated financial market environment...");
            
            // Create the environment
            var environment = new FinancialMarketEnvironment(
                numAssets: 3,                 // 3 assets (e.g., stocks, bonds, cash)
                initialPrices: new[] { 100.0, 50.0, 1.0 }, // Starting prices
                volatilities: new[] { 0.2, 0.1, 0.01 },    // Daily volatility for each asset
                correlations: new double[,] {              // Correlation matrix
                    { 1.0, 0.3, 0.0 },
                    { 0.3, 1.0, 0.0 },
                    { 0.0, 0.0, 1.0 }
                },
                tradingFee: 0.001,            // 0.1% trading fee
                initialBalance: 100000.0,     // Start with $100,000
                episodeLength: 252,           // 1 year of trading days
                includeMacroIndicators: true, // Include economic indicators
                includeTechnicalIndicators: true); // Include technical analysis indicators
            
            Console.WriteLine("Environment created with 3 assets and 1-year episodes.");
            
            return environment;
        }
        
        /// <summary>
        /// Trains the model on the simulated market environment.
        /// </summary>
        private static void TrainModel(HRARLModel<double> model, FinancialMarketEnvironment environment, int episodes)
        {
            Console.WriteLine($"\nTraining model for {episodes} episodes...");
            
            double totalReward = 0;
            double bestEpisodeReward = double.MinValue;
            double worstEpisodeReward = double.MaxValue;
            
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
                worstEpisodeReward = Math.Min(worstEpisodeReward, episodeReward);
                
                // Print progress
                if ((episode + 1) % 100 == 0 || episode == 0)
                {
                    double avgReward = totalReward / (episode + 1);
                    Console.WriteLine($"Episode {episode + 1}/{episodes} - Avg Reward: {avgReward:F2} - Best: {bestEpisodeReward:F2} - Worst: {worstEpisodeReward:F2}");
                }
            }
            
            Console.WriteLine("Training completed!");
            Console.WriteLine($"Final Average Reward: {totalReward / episodes:F2}");
            Console.WriteLine($"Best Episode Reward: {bestEpisodeReward:F2}");
            Console.WriteLine($"Worst Episode Reward: {worstEpisodeReward:F2}");
        }
        
        /// <summary>
        /// Evaluates the model with different risk aversion settings.
        /// </summary>
        private static void EvaluateModelWithDifferentRiskSettings(HRARLModel<double> model, FinancialMarketEnvironment environment)
        {
            Console.WriteLine("\nComparing performance with different risk aversion settings:");
            
            // Prepare for evaluation
            model.SetTrainingMode(false);
            
            var riskSettings = new[] { 0.1, 0.5, 0.9 }; // Low, medium, high risk aversion
            var descriptions = new[] { "Low Risk Aversion (Aggressive)", "Medium Risk Aversion (Balanced)", "High Risk Aversion (Conservative)" };
            var episodes = 100;
            
            for (int r = 0; r < riskSettings.Length; r++)
            {
                // Set risk aversion parameter
                model.SetRiskAversionParameter(riskSettings[r]);
                
                double totalReturn = 0;
                double maxDrawdown = 0;
                double sharpeRatio = 0;
                double winRate = 0;
                
                List<double> returns = new List<double>();
                
                // Run evaluation episodes
                for (int episode = 0; episode < episodes; episode++)
                {
                    var state = environment.Reset();
                    bool done = false;
                    double initialPortfolioValue = environment.GetPortfolioValue();
                    double episodeReturn = 0;
                    double highWaterMark = initialPortfolioValue;
                    double episodeDrawdown = 0;
                    bool isWin = false;
                    
                    List<double> dailyReturns = new List<double>();
                    double prevValue = initialPortfolioValue;
                    
                    while (!done)
                    {
                        var action = model.SelectAction(state, isTraining: false);
                        var (nextState, reward, isDone) = environment.Step(action);
                        
                        state = nextState;
                        episodeReturn += reward;
                        done = isDone;
                        
                        // Calculate drawdown
                        double currentValue = environment.GetPortfolioValue();
                        if (currentValue > highWaterMark)
                        {
                            highWaterMark = currentValue;
                        }
                        
                        double currentDrawdown = (highWaterMark - currentValue) / highWaterMark;
                        episodeDrawdown = Math.Max(episodeDrawdown, currentDrawdown);
                        
                        // Record daily return
                        double dailyReturn = (currentValue - prevValue) / prevValue;
                        dailyReturns.Add(dailyReturn);
                        prevValue = currentValue;
                    }
                    
                    double finalPortfolioValue = environment.GetPortfolioValue();
                    double totalEpisodeReturn = (finalPortfolioValue - initialPortfolioValue) / initialPortfolioValue;
                    
                    totalReturn += totalEpisodeReturn;
                    maxDrawdown += episodeDrawdown;
                    isWin = finalPortfolioValue > initialPortfolioValue;
                    winRate += isWin ? 1 : 0;
                    
                    // Calculate Sharpe ratio
                    double dailyReturnMean = dailyReturns.Average();
                    double dailyReturnStd = Math.Sqrt(dailyReturns.Select(r => Math.Pow(r - dailyReturnMean, 2)).Average());
                    double episodeSharpe = dailyReturnStd == 0 ? 0 : dailyReturnMean / dailyReturnStd;
                    sharpeRatio += episodeSharpe;
                    
                    returns.Add(totalEpisodeReturn);
                }
                
                // Calculate average metrics
                double avgReturn = totalReturn / episodes;
                double avgMaxDrawdown = maxDrawdown / episodes;
                double avgSharpeRatio = sharpeRatio / episodes;
                double avgWinRate = winRate / episodes * 100;
                
                // Calculate standard deviation of returns
                double returnMean = returns.Average();
                double returnStd = Math.Sqrt(returns.Select(r => Math.Pow(r - returnMean, 2)).Average());
                
                // Print results
                Console.WriteLine($"\n{descriptions[r]} - Risk Aversion = {riskSettings[r]:F1}");
                Console.WriteLine($"  Average Return: {avgReturn * 100:F2}%");
                Console.WriteLine($"  Average Max Drawdown: {avgMaxDrawdown * 100:F2}%");
                Console.WriteLine($"  Return Volatility: {returnStd * 100:F2}%");
                Console.WriteLine($"  Sharpe Ratio: {avgSharpeRatio:F2}");
                Console.WriteLine($"  Win Rate: {avgWinRate:F2}%");
            }
        }
        
        /// <summary>
        /// Demonstrates the risk analysis features of the HRARL model.
        /// </summary>
        private static void DemonstrateRiskAnalysis(HRARLModel<double> model, FinancialMarketEnvironment environment)
        {
            Console.WriteLine("\nRisk Analysis for Different Trading Actions:");
            
            // Get a sample market state
            var state = environment.Reset();
            
            // Analyze risks for different potential actions
            var riskAnalysis = model.AnalyzeActionRisks(state);
            
            // Print results in a sorted table (by risk-adjusted return)
            Console.WriteLine("\n{0,-25} {1,-15} {2,-15} {3,-15} {4,-15} {5,-15}", 
                "Action", "Exp. Return", "CVaR (95%)", "Risk-Adj Ret", "Win Prob", "Return Var");
            Console.WriteLine(new string('-', 100));
            
            foreach (var entry in riskAnalysis.OrderByDescending(e => e.Value.RiskAdjustedReturn))
            {
                Console.WriteLine("{0,-25} {1,15:F4} {2,15:F4} {3,15:F4} {4,14:F2}% {5,15:F4}",
                    entry.Key,
                    entry.Value.ExpectedReturn,
                    entry.Value.ConditionalValueAtRisk95,
                    entry.Value.RiskAdjustedReturn,
                    entry.Value.ProbabilityOfPositiveReturn * 100,
                    entry.Value.ReturnVariance);
            }
            
            // Demonstrate hierarchical nature of the model
            Console.WriteLine("\nDemonstrating Hierarchical Decision Making:");
            var highLevelGoal = model.GetHighLevelGoal(state);
            
            Console.WriteLine("\nHigh-Level Strategic Goal:");
            if (highLevelGoal.Length >= 3)
            {
                Console.WriteLine($"  Asset Allocation Target: {InterpretAllocation(highLevelGoal)}");
                Console.WriteLine($"  Risk Budget: {(highLevelGoal[highLevelGoal.Length - 1] * 100):F2}%");
                Console.WriteLine($"  Current Market Risk Assessment: {model.CurrentRiskAssessment * 100:F2}%");
            }
            else
            {
                for (int i = 0; i < highLevelGoal.Length; i++)
                {
                    Console.WriteLine($"  Component {i+1}: {highLevelGoal[i]:F4}");
                }
            }
        }
        
        /// <summary>
        /// Demonstrates the strategy simulation features of the HRARL model.
        /// </summary>
        private static void DemonstrateStrategySimulation(HRARLModel<double> model, FinancialMarketEnvironment environment)
        {
            Console.WriteLine("\nSimulating Trading Strategy Performance:");
            
            // Get a sample market state
            var state = environment.Reset();
            
            // Simulate the strategy over multiple time horizons and risk settings
            int numSteps = 60; // 60 trading days (about 3 months)
            int numSimulations = 100;
            
            // Set different risk aversion parameters
            var riskSettings = new[] { 0.1, 0.5, 0.9 }; // Low, medium, high risk aversion
            var descriptions = new[] { "Low Risk Aversion (Aggressive)", "Medium Risk Aversion (Balanced)", "High Risk Aversion (Conservative)" };
            
            for (int r = 0; r < riskSettings.Length; r++)
            {
                // Set risk aversion parameter
                model.SetRiskAversionParameter(riskSettings[r]);
                
                // Simulate strategy
                var simulationResults = model.SimulateStrategy(state, numSteps, numSimulations);
                
                // Calculate statistics
                var finalReturns = simulationResults.Select(trajectory => trajectory[numSteps - 1]).ToList();
                double meanReturn = finalReturns.Average();
                double stdReturn = Math.Sqrt(finalReturns.Select(r => Math.Pow(r - meanReturn, 2)).Average());
                double maxReturn = finalReturns.Max();
                double minReturn = finalReturns.Min();
                
                // Calculate VaR and CVaR
                finalReturns.Sort();
                int varIndex = (int)(numSimulations * 0.05); // 5% VaR
                double var95 = finalReturns[varIndex];
                double cvar95 = finalReturns.Take(varIndex + 1).Average();
                
                // Calculate probability of positive return
                double probPositive = finalReturns.Count(r => r > 0) / (double)numSimulations * 100;
                
                // Print results
                Console.WriteLine($"\n{descriptions[r]} - Risk Aversion = {riskSettings[r]:F1}");
                Console.WriteLine($"  Mean Return: {meanReturn:F2}");
                Console.WriteLine($"  Return Std Dev: {stdReturn:F2}");
                Console.WriteLine($"  Max Return: {maxReturn:F2}");
                Console.WriteLine($"  Min Return: {minReturn:F2}");
                Console.WriteLine($"  5% VaR: {var95:F2}");
                Console.WriteLine($"  5% CVaR: {cvar95:F2}");
                Console.WriteLine($"  Probability of Positive Return: {probPositive:F2}%");
            }
        }
        
        /// <summary>
        /// Interprets a goal vector as an asset allocation strategy.
        /// </summary>
        private static string InterpretAllocation(Vector<double> goal)
        {
            if (goal.Length < 3)
                return "N/A";
            
            // Normalize the first components to represent asset allocation percentages
            double sum = 0;
            for (int i = 0; i < 3; i++) // Assuming 3 assets
            {
                sum += Math.Abs(goal[i]);
            }
            
            if (sum == 0)
                return "Cash: 100.00%";
            
            string[] assetNames = { "Stocks", "Bonds", "Cash" };
            var allocation = new List<string>();
            
            for (int i = 0; i < 3; i++)
            {
                double percentage = Math.Abs(goal[i]) / sum * 100;
                allocation.Add($"{assetNames[i]}: {percentage:F2}%");
            }
            
            return string.Join(", ", allocation);
        }
        
        /// <summary>
        /// A simulated financial market environment for testing reinforcement learning models.
        /// </summary>
        private class FinancialMarketEnvironment
        {
            private readonly int _numAssets;
            private readonly double[] _initialPrices;
            private readonly double[] _volatilities;
            private readonly double[,] _correlations;
            private readonly double _tradingFee;
            private readonly int _episodeLength;
            private readonly bool _includeMacroIndicators;
            private readonly bool _includeTechnicalIndicators;
            private readonly Random _random = default!;
            
            private double[] _currentPrices;
            private double[] _positions;
            private double _cash;
            private int _currentDay;
            private double _initialBalance;
            
            // Macro economic indicators (simplified for simulation)
            private double _interestRate;
            private double _inflationRate;
            private double _economicGrowth;
            private double _marketSentiment;
            
            // Technical indicators (simplified for simulation)
            private double[] _movingAverages;
            private double[] _rsi;
            private double[] _volatilityIndicator;
            
            /// <summary>
            /// Creates a new instance of the FinancialMarketEnvironment class.
            /// </summary>
            public FinancialMarketEnvironment(
                int numAssets,
                double[] initialPrices,
                double[] volatilities,
                double[,] correlations,
                double tradingFee,
                double initialBalance,
                int episodeLength,
                bool includeMacroIndicators,
                bool includeTechnicalIndicators)
            {
                _numAssets = numAssets;
                _initialPrices = initialPrices;
                _volatilities = volatilities;
                _correlations = correlations;
                _tradingFee = tradingFee;
                _initialBalance = initialBalance;
                _episodeLength = episodeLength;
                _includeMacroIndicators = includeMacroIndicators;
                _includeTechnicalIndicators = includeTechnicalIndicators;
                _random = new Random(42); // Fixed seed for reproducibility
                
                _currentPrices = new double[_numAssets];
                _positions = new double[_numAssets];
                
                // Initialize technical indicators
                _movingAverages = new double[_numAssets];
                _rsi = new double[_numAssets];
                _volatilityIndicator = new double[_numAssets];
                
                Reset();
            }
            
            /// <summary>
            /// Resets the environment to start a new episode.
            /// </summary>
            public Tensor<double> Reset()
            {
                // Reset prices
                Array.Copy(_initialPrices, _currentPrices, _numAssets);
                
                // Reset positions
                Array.Clear(_positions, 0, _positions.Length);
                
                // Reset cash
                _cash = _initialBalance;
                
                // Reset day counter
                _currentDay = 0;
                
                // Reset macro indicators
                _interestRate = 0.02 + 0.01 * (_random.NextDouble() - 0.5); // 1.5% - 2.5%
                _inflationRate = 0.02 + 0.01 * (_random.NextDouble() - 0.5); // 1.5% - 2.5%
                _economicGrowth = 0.025 + 0.015 * (_random.NextDouble() - 0.5); // 1% - 4%
                _marketSentiment = 0.5 + 0.3 * (_random.NextDouble() - 0.5); // 0.2 - 0.8
                
                // Reset technical indicators
                UpdateTechnicalIndicators();
                
                // Return the initial state
                return GetState();
            }
            
            /// <summary>
            /// Takes a step in the environment based on the given action.
            /// </summary>
            public (Tensor<double>, double, bool) Step(Vector<double> action)
            {
                // Store initial portfolio value
                double initialPortfolioValue = GetPortfolioValue();
                
                // Execute trading action
                ExecuteAction(action);
                
                // Update market prices and indicators
                UpdatePrices();
                UpdateMacroIndicators();
                UpdateTechnicalIndicators();
                
                // Increment day counter
                _currentDay++;
                
                // Calculate reward as the change in portfolio value
                double finalPortfolioValue = GetPortfolioValue();
                double reward = (finalPortfolioValue - initialPortfolioValue) / initialPortfolioValue;
                
                // Check if episode is done
                bool isDone = _currentDay >= _episodeLength || finalPortfolioValue <= 0;
                
                // Return new state, reward, and done flag
                return (GetState(), reward, isDone);
            }
            
            /// <summary>
            /// Gets the current portfolio value.
            /// </summary>
            public double GetPortfolioValue()
            {
                double value = _cash;
                
                for (int i = 0; i < _numAssets; i++)
                {
                    value += _positions[i] * _currentPrices[i];
                }
                
                return value;
            }
            
            /// <summary>
            /// Gets the current state representation.
            /// </summary>
            private Tensor<double> GetState()
            {
                // Calculate the state dimension based on enabled features
                int priceFeaturesCount = _numAssets * 2; // Prices and normalized prices
                int positionFeaturesCount = _numAssets + 1; // Asset positions and cash
                int macroFeaturesCount = _includeMacroIndicators ? 4 : 0; // Interest rate, inflation, growth, sentiment
                int technicalFeaturesCount = _includeTechnicalIndicators ? _numAssets * 3 : 0; // MA, RSI, volatility
                
                int stateDimension = priceFeaturesCount + positionFeaturesCount + macroFeaturesCount + technicalFeaturesCount;
                
                var state = new Tensor<double>(new[] { 1, stateDimension });
                int featureIdx = 0;
                
                // Add current absolute prices
                for (int i = 0; i < _numAssets; i++)
                {
                    state[0, featureIdx++] = _currentPrices[i];
                }
                
                // Add normalized prices (% change from initial)
                for (int i = 0; i < _numAssets; i++)
                {
                    double normalizedPrice = _currentPrices[i] / _initialPrices[i] - 1.0;
                    state[0, featureIdx++] = normalizedPrice;
                }
                
                // Add current positions
                for (int i = 0; i < _numAssets; i++)
                {
                    state[0, featureIdx++] = _positions[i];
                }
                
                // Add cash position
                state[0, featureIdx++] = _cash;
                
                // Add macro indicators if enabled
                if (_includeMacroIndicators)
                {
                    state[0, featureIdx++] = _interestRate;
                    state[0, featureIdx++] = _inflationRate;
                    state[0, featureIdx++] = _economicGrowth;
                    state[0, featureIdx++] = _marketSentiment;
                }
                
                // Add technical indicators if enabled
                if (_includeTechnicalIndicators)
                {
                    for (int i = 0; i < _numAssets; i++)
                    {
                        state[0, featureIdx++] = _movingAverages[i];
                    }
                    
                    for (int i = 0; i < _numAssets; i++)
                    {
                        state[0, featureIdx++] = _rsi[i];
                    }
                    
                    for (int i = 0; i < _numAssets; i++)
                    {
                        state[0, featureIdx++] = _volatilityIndicator[i];
                    }
                }
                
                return state;
            }
            
            /// <summary>
            /// Executes a trading action.
            /// </summary>
            private void ExecuteAction(Vector<double> action)
            {
                double portfolioValue = GetPortfolioValue();
                
                // Interpret action as target asset allocations
                double[] targetAllocations = new double[_numAssets];
                double remainingAllocation = 1.0;
                
                // Process all but the last asset
                for (int i = 0; i < _numAssets - 1 && i < action.Length; i++)
                {
                    // Map from [-1, 1] to [0, remainingAllocation]
                    targetAllocations[i] = (action[i] + 1.0) / 2.0 * remainingAllocation;
                    remainingAllocation -= targetAllocations[i];
                }
                
                // Allocate any remaining to the last asset
                targetAllocations[_numAssets - 1] = remainingAllocation;
                
                // Calculate target positions in units
                double[] targetPositions = new double[_numAssets];
                for (int i = 0; i < _numAssets; i++)
                {
                    targetPositions[i] = targetAllocations[i] * portfolioValue / _currentPrices[i];
                }
                
                // Execute trades
                for (int i = 0; i < _numAssets; i++)
                {
                    if (Math.Abs(_positions[i] - targetPositions[i]) > 0.001) // Avoid tiny trades
                    {
                        double sharesDelta = targetPositions[i] - _positions[i];
                        double transactionValue = sharesDelta * _currentPrices[i];
                        double fee = Math.Abs(transactionValue) * _tradingFee;
                        
                        // Update positions and cash
                        _positions[i] += sharesDelta;
                        _cash -= (transactionValue + fee);
                    }
                }
            }
            
            /// <summary>
            /// Updates asset prices for the next day.
            /// </summary>
            private void UpdatePrices()
            {
                // Generate correlated random returns
                double[] assetReturns = GenerateCorrelatedReturns();
                
                // Update prices based on returns
                for (int i = 0; i < _numAssets; i++)
                {
                    _currentPrices[i] *= (1.0 + assetReturns[i]);
                    
                    // Apply macro effects
                    _currentPrices[i] *= (1.0 + (_economicGrowth - _inflationRate) * 0.01);
                    
                    // Apply sentiment effect
                    double sentimentEffect = (_marketSentiment - 0.5) * 0.005;
                    _currentPrices[i] *= (1.0 + sentimentEffect);
                    
                    // Ensure price doesn't go below a minimum value
                    _currentPrices[i] = Math.Max(_currentPrices[i], 0.01);
                }
            }
            
            /// <summary>
            /// Updates macro-economic indicators.
            /// </summary>
            private void UpdateMacroIndicators()
            {
                // Interest rate changes slowly, mean-reverts to around 2%
                _interestRate += 0.001 * (_random.NextDouble() - 0.5);
                _interestRate = 0.02 + 0.5 * (_interestRate - 0.02); // Mean reversion
                _interestRate = Math.Max(0.001, Math.Min(0.05, _interestRate)); // Bounds
                
                // Inflation rate changes slowly
                _inflationRate += 0.001 * (_random.NextDouble() - 0.5);
                _inflationRate = 0.02 + 0.7 * (_inflationRate - 0.02); // Mean reversion
                _inflationRate = Math.Max(0.0, Math.Min(0.05, _inflationRate)); // Bounds
                
                // Economic growth changes with some momentum
                _economicGrowth += 0.002 * (_random.NextDouble() - 0.5);
                _economicGrowth = 0.025 + 0.6 * (_economicGrowth - 0.025); // Mean reversion
                _economicGrowth = Math.Max(-0.02, Math.Min(0.06, _economicGrowth)); // Bounds
                
                // Market sentiment is more volatile
                _marketSentiment += 0.05 * (_random.NextDouble() - 0.5);
                _marketSentiment = 0.5 + 0.3 * (_marketSentiment - 0.5); // Mean reversion
                _marketSentiment = Math.Max(0.0, Math.Min(1.0, _marketSentiment)); // Bounds
            }
            
            /// <summary>
            /// Updates technical indicators based on the current prices.
            /// </summary>
            private void UpdateTechnicalIndicators()
            {
                for (int i = 0; i < _numAssets; i++)
                {
                    // Simple moving average, trending towards the current price
                    if (_currentDay == 0)
                    {
                        _movingAverages[i] = _currentPrices[i];
                    }
                    else
                    {
                        _movingAverages[i] = 0.9 * _movingAverages[i] + 0.1 * _currentPrices[i];
                    }
                    
                    // Normalized MA indicator (-1 to 1)
                    _movingAverages[i] = (_currentPrices[i] / _movingAverages[i]) - 1.0;
                    
                    // Simplified RSI (0 to 1)
                    _rsi[i] = 0.7 * _rsi[i] + 0.3 * _random.NextDouble();
                    
                    // Simplified volatility indicator
                    _volatilityIndicator[i] = _volatilities[i] * (0.8 + 0.4 * _random.NextDouble());
                }
            }
            
            /// <summary>
            /// Generates correlated asset returns based on the correlation matrix.
            /// </summary>
            private double[] GenerateCorrelatedReturns()
            {
                // Generate independent standard normal random variables
                double[] independentReturns = new double[_numAssets];
                for (int i = 0; i < _numAssets; i++)
                {
                    independentReturns[i] = SampleGaussian();
                }
                
                // Apply Cholesky decomposition of correlation matrix (simplified)
                double[] correlatedReturns = new double[_numAssets];
                for (int i = 0; i < _numAssets; i++)
                {
                    correlatedReturns[i] = 0;
                    for (int j = 0; j <= i; j++)
                    {
                        correlatedReturns[i] += _correlations[i, j] * independentReturns[j];
                    }
                    
                    // Scale by volatility and convert to return
                    correlatedReturns[i] = correlatedReturns[i] * _volatilities[i];
                }
                
                return correlatedReturns;
            }
            
            /// <summary>
            /// Samples a value from a standard Gaussian distribution.
            /// </summary>
            private double SampleGaussian()
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            }
        }
    }
}