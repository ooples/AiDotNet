using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Models;
using AiDotNet.ReinforcementLearning.Models.Options;
using System;
using System.Collections.Generic;
using System.IO;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating the use of Model-Based Policy Optimization (MBPO) for stock market prediction and trading.
    /// </summary>
    public class MBPOExample
    {
        /// <summary>
        /// Runs the MBPO example for stock trading.
        /// </summary>
        public static void RunExample()
        {
            Console.WriteLine("Model-Based Policy Optimization Example for Stock Trading");
            Console.WriteLine("=====================================================");
            
            // 1. Configure the MBPO model
            var options = CreateMBPOOptions();
            
            // 2. Create the model
            var model = new MBPOModel<double>(options);
            Console.WriteLine("MBPO model created successfully.");
            
            // 3. Create a simple stock market environment
            var environment = new SimpleStockEnvironment(1000);
            Console.WriteLine($"Created stock market environment with initial price: $1000.00");
            
            // 4. Train the model using both real data and model-generated data
            Console.WriteLine("\nTraining the model (hybrid approach with real and simulated data)...");
            TrainModel(model, environment, 100);
            
            // 5. Display model training statistics
            DisplayModelStats(model);
            
            // 6. Evaluate the trained model
            Console.WriteLine("\nEvaluating the trained model...");
            EvaluateModel(model, environment, 20);
            
            // 7. Demonstrate future state prediction using the learned dynamics model
            Console.WriteLine("\nDemonstrating future state prediction...");
            DemonstrateFuturePrediction(model, environment);
            
            Console.WriteLine("\nModel-Based Policy Optimization Example completed successfully!");
        }
        
        /// <summary>
        /// Creates the options for the MBPO algorithm.
        /// </summary>
        /// <returns>The configured options.</returns>
        private static MBPOOptions CreateMBPOOptions()
        {
            return new MBPOOptions
            {
                // State and action dimensions
                StateSize = 5,      // Market features (price, indicators, etc.)
                ActionSize = 3,     // Actions (buy, hold, sell)
                IsContinuous = false, // Discrete actions
                
                // MBPO specific parameters
                EnsembleSize = 5,          // Number of dynamics models in the ensemble
                ModelRatio = 5,            // Ratio of model data to real data (lower for this example)
                RolloutHorizon = 1,        // How many steps to roll out
                // ModelTrainingFrequency is not a property - model training is handled internally
                
                // Neural network architecture
                ModelHiddenSizes = new int[] { 64, 64 },   // Dynamics model
                PolicyHiddenSizes = new int[] { 64, 64 },  // Policy network
                ValueHiddenSizes = new int[] { 64, 64 },   // Value network
                
                // Learning parameters
                ModelLearningRate = 0.001,  // Dynamics model learning rate
                PolicyLearningRate = 0.0003, // Policy learning rate
                ValueLearningRate = 0.0003,  // Value function learning rate
                
                // Model settings
                ProbabilisticModel = true,   // Use probabilistic dynamics model
                ModelPredictRewards = true,  // Model predicts rewards too
                
                // General RL parameters
                Gamma = 0.95,           // Discount factor
                BatchSize = 32,         // Batch size for updates
                RealExpBeforeModel = 500, // Real experiences before using model
                
                // SAC parameters
                InitialTemperature = 0.1, // Initial entropy coefficient
                AutoTuneEntropy = true,   // Automatically tune entropy
                
                // Advanced rollout settings
                BranchingRollouts = false, // Don't use branching for simplicity
                NumBranches = 2,          // Number of branches if using branching
                
                // Buffer sizes
                ReplayBufferCapacity = 10000,    // Experience buffer size
                
                // Training parameters
                ModelEpochs = 10,        // Epochs for model training
                PolicyEpochs = 5         // Epochs for policy training
            };
        }
        
        /// <summary>
        /// Trains the MBPO model by interacting with the environment.
        /// </summary>
        /// <param name="model">The MBPO model to train.</param>
        /// <param name="environment">The stock environment.</param>
        /// <param name="episodes">The number of episodes to train for.</param>
        private static void TrainModel(MBPOModel<double> model, SimpleStockEnvironment environment, int episodes)
        {
            model.SetTrainingMode();
            
            double totalReward = 0;
            
            for (int episode = 0; episode < episodes; episode++)
            {
                // Reset environment for new episode
                environment.Reset();
                bool done = false;
                double episodeReward = 0;
                
                // Get initial state
                var state = environment.GetCurrentState();
                
                // Episode loop
                while (!done)
                {
                    // Select action using model
                    var action = model.SelectAction(state, true);
                    
                    // Take action in environment
                    var (nextState, reward, done_) = environment.Step(action);
                    done = done_;
                    
                    // Update model based on experience (real data)
                    model.Update(state, action, reward, nextState, done);
                    
                    // Accumulate reward
                    episodeReward += reward;
                    
                    // Update state
                    state = nextState;
                }
                
                totalReward += episodeReward;
                
                // Print progress every few episodes
                if (episode % 10 == 0 || episode == episodes - 1)
                {
                    double averageReward = totalReward / (episode + 1);
                    Console.WriteLine($"Episode {episode + 1}/{episodes}, Average Reward: {averageReward:F2}, Current Portfolio: ${environment.CurrentCapital:F2}");
                    
                    // Display model stats
                    if (episode > 0 && episode % 25 == 0)
                    {
                        DisplayModelStats(model);
                    }
                }
            }
            
            Console.WriteLine($"Training completed over {episodes} episodes with final average reward: {totalReward / episodes:F2}");
        }
        
        /// <summary>
        /// Displays statistics about the model's training process.
        /// </summary>
        /// <param name="model">The trained MBPO model.</param>
        private static void DisplayModelStats(MBPOModel<double> model)
        {
            var stats = model.GetTrainingStats();
            
            Console.WriteLine("\nModel Training Statistics:");
            Console.WriteLine($"  Real Experiences: {stats["TotalRealExperiences"]}");
            Console.WriteLine($"  Model-Generated Experiences: {stats["TotalModelExperiences"]}");
            Console.WriteLine($"  Model Training Iterations: {stats["ModelTrainingIterations"]}");
            Console.WriteLine($"  Real-to-Synthetic Ratio: 1:{stats["RealToSyntheticRatio"]:F1}");
            
            // Display losses if available
            if (stats.ContainsKey("PolicyLoss"))
                Console.WriteLine($"  Policy Loss: {stats["PolicyLoss"]:F4}");
            if (stats.ContainsKey("ValueLoss"))
                Console.WriteLine($"  Value Loss: {stats["ValueLoss"]:F4}");
            if (stats.ContainsKey("ModelLoss"))
                Console.WriteLine($"  Model Loss: {stats["ModelLoss"]:F4}");
        }
        
        /// <summary>
        /// Evaluates the model on the environment.
        /// </summary>
        /// <param name="model">The MBPO model to evaluate.</param>
        /// <param name="environment">The stock environment.</param>
        /// <param name="episodes">The number of episodes to evaluate for.</param>
        private static void EvaluateModel(MBPOModel<double> model, SimpleStockEnvironment environment, int episodes)
        {
            model.SetEvaluationMode();
            
            double totalReward = 0;
            int totalTrades = 0;
            int profitableTrades = 0;
            double maxDrawdown = 0;
            double peakValue = environment.InitialCapital;
            
            for (int episode = 0; episode < episodes; episode++)
            {
                // Reset environment for new episode
                environment.Reset();
                bool done = false;
                double episodeReward = 0;
                double episodeInitialCapital = environment.CurrentCapital;
                
                // Get initial state
                var state = environment.GetCurrentState();
                
                // Track trades and capital in this episode
                List<string> tradeLog = new List<string>();
                
                // Episode loop
                int step = 0;
                while (!done)
                {
                    step++;
                    
                    // Select action using model
                    var action = model.SelectAction(state, false);
                    
                    // Log the action and current state
                    int actionIndex = GetActionIndex(action);
                    string actionName = actionIndex == 0 ? "BUY" : (actionIndex == 1 ? "HOLD" : "SELL");
                    
                    // Take action in environment
                    double priorCapital = environment.CurrentCapital;
                    var (nextState, reward, done_) = environment.Step(action);
                    done = done_;
                    
                    // Update tracking metrics
                    episodeReward += reward;
                    double currentValue = environment.CurrentCapital;
                    
                    // Track peak and drawdown
                    if (currentValue > peakValue)
                    {
                        peakValue = currentValue;
                    }
                    else
                    {
                        double currentDrawdown = (peakValue - currentValue) / peakValue;
                        if (currentDrawdown > maxDrawdown)
                        {
                            maxDrawdown = currentDrawdown;
                        }
                    }
                    
                    // Track trades
                    if (actionIndex == 0 || actionIndex == 2)  // Buy or Sell
                    {
                        totalTrades++;
                        double tradePnL = currentValue - priorCapital;
                        
                        if (tradePnL > 0)
                        {
                            profitableTrades++;
                        }
                        
                        tradeLog.Add($"Step {step}: {actionName} - Price: ${environment.CurrentPrice:F2}, Capital: ${currentValue:F2}, P&L: ${tradePnL:F2}");
                    }
                    
                    // Update state
                    state = nextState;
                }
                
                totalReward += episodeReward;
                double episodeReturn = (environment.CurrentCapital - episodeInitialCapital) / episodeInitialCapital * 100;
                
                // Print episode result
                Console.WriteLine($"Episode {episode + 1}, Return: {episodeReturn:F2}%, Final Capital: ${environment.CurrentCapital:F2}");
                
                // Show detailed trade log for a few episodes
                if (episode < 3 && tradeLog.Count > 0)
                {
                    Console.WriteLine("  Trade Log:");
                    foreach (var trade in tradeLog)
                    {
                        Console.WriteLine($"  {trade}");
                    }
                    Console.WriteLine();
                }
            }
            
            // Print overall statistics
            double averageReward = totalReward / episodes;
            double winRate = totalTrades > 0 ? (double)profitableTrades / totalTrades * 100 : 0;
            
            Console.WriteLine("\nEvaluation Results:");
            Console.WriteLine($"  Average Reward: {averageReward:F2}");
            Console.WriteLine($"  Win Rate: {winRate:F2}% ({profitableTrades}/{totalTrades} trades)");
            Console.WriteLine($"  Maximum Drawdown: {maxDrawdown * 100:F2}%");
            Console.WriteLine($"  Risk-Adjusted Return: {(averageReward / (maxDrawdown > 0 ? maxDrawdown : 0.01)):F2}");
        }
        
        /// <summary>
        /// Demonstrates the model's ability to predict future states.
        /// </summary>
        /// <param name="model">The trained MBPO model.</param>
        /// <param name="environment">The stock environment.</param>
        private static void DemonstrateFuturePrediction(MBPOModel<double> model, SimpleStockEnvironment environment)
        {
            // Reset environment and get current state
            environment.Reset();
            var currentState = environment.GetCurrentState();
            
            Console.WriteLine("\nCurrent Market State:");
            Console.WriteLine($"  Price: ${environment.CurrentPrice:F2}");
            Console.WriteLine($"  Normalized Price: {currentState[0]:F4}");
            Console.WriteLine($"  Momentum: {currentState[1]:F4}");
            Console.WriteLine($"  Volatility: {currentState[2]:F4}");
            
            // Get the recommended action
            var action = model.SelectAction(currentState, false);
            int actionIndex = GetActionIndex(action);
            string actionName = actionIndex == 0 ? "BUY" : (actionIndex == 1 ? "HOLD" : "SELL");
            
            Console.WriteLine($"\nRecommended Action: {actionName}");
            
            // Get prediction uncertainty
            double uncertainty = model.GetPredictionUncertainty(currentState, action);
            Console.WriteLine($"Prediction Uncertainty: {uncertainty:F4}");
            
            // Predict future states
            int stepsToPredict = 5;
            Console.WriteLine($"\nPredicting Market Evolution for {stepsToPredict} Steps:");
            
            var futurePredictions = model.PredictFutureStates(currentState, action, stepsToPredict);
            
            // Original price
            double basePrice = environment.CurrentPrice;
            
            for (int i = 1; i < futurePredictions.Count; i++) // Skip the first one which is the current state
            {
                // Extract predicted price factor and calculate actual price
                double priceFactor = futurePredictions[i][0];
                double predictedPrice = basePrice * priceFactor;
                
                // Extract other predictions
                double predictedMomentum = futurePredictions[i][1];
                double predictedVolatility = futurePredictions[i][2];
                
                Console.WriteLine($"Step {i}:");
                Console.WriteLine($"  Predicted Price: ${predictedPrice:F2} ({(priceFactor > 1 ? "+" : "")}{(priceFactor - 1) * 100:F2}%)");
                Console.WriteLine($"  Predicted Momentum: {predictedMomentum:F4}");
                Console.WriteLine($"  Predicted Volatility: {predictedVolatility:F4}");
                
                // Make a new action recommendation for this predicted state
                var nextAction = model.SelectAction(futurePredictions[i], false);
                int nextActionIndex = GetActionIndex(nextAction);
                string nextActionName = nextActionIndex == 0 ? "BUY" : (nextActionIndex == 1 ? "HOLD" : "SELL");
                
                Console.WriteLine($"  Recommended Action: {nextActionName}");
            }
        }
        
        /// <summary>
        /// Gets the index of the selected action from an action vector.
        /// </summary>
        /// <param name="action">The action vector.</param>
        /// <returns>The index of the selected action.</returns>
        private static int GetActionIndex(Vector<double> action)
        {
            int bestAction = 0;
            double bestValue = action[0];
            
            for (int i = 1; i < action.Length; i++)
            {
                if (action[i] > bestValue)
                {
                    bestValue = action[i];
                    bestAction = i;
                }
            }
            
            return bestAction;
        }
    }
    
    /// <summary>
    /// A simple stock market environment for reinforcement learning.
    /// </summary>
    public class SimpleStockEnvironment
    {
        private readonly Random _random = default!;
        private readonly double _initialPrice;
        private readonly int _maxSteps;
        
        private double _currentPrice;
        private double _cash;
        private double _shares;
        private int _steps;
        private double _momentum;
        private double _volatility;
        private bool _inPosition;
        
        /// <summary>
        /// Gets the initial capital (cash + shares value).
        /// </summary>
        public double InitialCapital { get; }
        
        /// <summary>
        /// Gets the current capital (cash + shares value).
        /// </summary>
        public double CurrentCapital => _cash + _shares * _currentPrice;
        
        /// <summary>
        /// Gets the current stock price.
        /// </summary>
        public double CurrentPrice => _currentPrice;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="SimpleStockEnvironment"/> class.
        /// </summary>
        /// <param name="initialPrice">The initial stock price.</param>
        public SimpleStockEnvironment(double initialPrice)
        {
            _random = new Random(42);
            _initialPrice = initialPrice;
            InitialCapital = 10000.0;
            _maxSteps = 20;
            
            Reset();
        }
        
        /// <summary>
        /// Resets the environment to its initial state.
        /// </summary>
        public void Reset()
        {
            _currentPrice = _initialPrice;
            _cash = InitialCapital;
            _shares = 0;
            _steps = 0;
            _momentum = 0;
            _volatility = 0.01;
            _inPosition = false;
        }
        
        /// <summary>
        /// Gets the current state of the environment.
        /// </summary>
        /// <returns>A tensor representing the current state.</returns>
        public Tensor<double> GetCurrentState()
        {
            // Create state vector with relevant features
            var state = new Vector<double>(5);
            
            // Feature 1: Normalized price
            state[0] = _currentPrice / _initialPrice;
            
            // Feature 2: Normalized momentum (price direction)
            state[1] = _momentum;
            
            // Feature 3: Normalized volatility
            state[2] = _volatility;
            
            // Feature 4: Position indicator (in position or not)
            state[3] = _inPosition ? 1.0 : 0.0;
            
            // Feature 5: Normalized portfolio value
            state[4] = CurrentCapital / InitialCapital;
            
            return Tensor<double>.FromVector(state);
        }
        
        /// <summary>
        /// Takes a step in the environment based on the given action.
        /// </summary>
        /// <param name="action">The action to take (buy, hold, sell).</param>
        /// <returns>The next state, reward, and whether the episode is done.</returns>
        public (Tensor<double>, double, bool) Step(Vector<double> action)
        {
            // Interpret the action (one-hot encoded)
            int actionIndex = 1; // Default to hold
            double maxValue = action[0];
            
            for (int i = 1; i < action.Length; i++)
            {
                if (action[i] > maxValue)
                {
                    maxValue = action[i];
                    actionIndex = i;
                }
            }
            
            // Execute the action
            double reward = 0;
            double initialPortfolioValue = CurrentCapital;
            
            switch (actionIndex)
            {
                case 0: // Buy
                    if (!_inPosition)
                    {
                        // Buy with all available cash
                        double sharesToBuy = _cash / _currentPrice;
                        _shares += sharesToBuy;
                        _cash = 0;
                        _inPosition = true;
                        reward = -0.001; // Small transaction cost
                    }
                    else
                    {
                        reward = -0.001; // Penalty for invalid action
                    }
                    break;
                    
                case 1: // Hold
                    // No action
                    break;
                    
                case 2: // Sell
                    if (_inPosition)
                    {
                        // Sell all shares
                        _cash += _shares * _currentPrice;
                        
                        // Calculate profit/loss as percentage of portfolio value
                        double portfolioValue = CurrentCapital;
                        double pnl = (portfolioValue - initialPortfolioValue) / initialPortfolioValue;
                        reward = pnl;
                        
                        _shares = 0;
                        _inPosition = false;
                    }
                    else
                    {
                        reward = -0.001; // Penalty for invalid action
                    }
                    break;
            }
            
            // Update the price for the next step
            UpdatePrice();
            
            // If we're holding a position, calculate unrealized PnL
            if (_inPosition && actionIndex == 1)
            {
                double portfolioValue = CurrentCapital;
                double pnl = (portfolioValue - initialPortfolioValue) / initialPortfolioValue;
                reward = pnl * 0.1; // Smaller reward for unrealized gains
            }
            
            // Check if the episode is done
            _steps++;
            bool done = _steps >= _maxSteps;
            
            // If the episode is ending and we're still in a position, sell everything
            if (done && _inPosition)
            {
                _cash += _shares * _currentPrice;
                
                // Calculate final profit/loss
                double portfolioValue = CurrentCapital;
                double pnl = (portfolioValue - initialPortfolioValue) / initialPortfolioValue;
                reward = pnl;
                
                _shares = 0;
                _inPosition = false;
            }
            
            // Return the next state, reward, and done flag
            return (GetCurrentState(), reward, done);
        }
        
        /// <summary>
        /// Updates the stock price for the next step.
        /// </summary>
        private void UpdatePrice()
        {
            // Occasionally shift market regime
            if (_random.NextDouble() < 0.05)
            {
                _momentum = (_random.NextDouble() - 0.5) * 0.02;
                _volatility = 0.005 + _random.NextDouble() * 0.015;
            }
            
            // Generate return
            double noise = (_random.NextDouble() - 0.5) * _volatility;
            double dailyReturn = _momentum + noise;
            
            // Update price
            _currentPrice *= (1 + dailyReturn);
            
            // Ensure price doesn't go too low
            _currentPrice = Math.Max(_currentPrice, _initialPrice * 0.1);
        }
    }
}