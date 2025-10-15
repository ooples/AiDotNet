using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Models;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.Memory;
using System;
using System.IO;
using System.Collections.Generic;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating the use of Quantile Regression DQN (QR-DQN) for stock market prediction and trading.
    /// </summary>
    public class QRDQNExample
    {
        /// <summary>
        /// Runs the QR-DQN example for stock trading.
        /// </summary>
        public static void RunExample()
        {
            Console.WriteLine("Quantile Regression DQN Example for Stock Trading");
            Console.WriteLine("================================================");
            
            // 1. Configure the QR-DQN model
            var options = CreateQRDQNOptions();
            
            // 2. Create the model
            var model = new QRDQNModel<double>(options);
            Console.WriteLine("QR-DQN model created successfully.");
            
            // 3. Generate sample data for a simple stock environment
            var environment = new QRDQNStockEnvironment(1000);
            Console.WriteLine("Created stock market environment with initial price: $1000.00");
            
            // 4. Train the model through environment interaction
            Console.WriteLine("\nTraining the model through environment interaction...");
            TrainModel(model, environment, 100);
            
            // 5. Evaluate the model and demonstrate risk-aware decision making
            Console.WriteLine("\nEvaluating the model with different risk settings...");
            
            // Standard evaluation (expected value)
            Console.WriteLine("\nStandard Strategy (Expected Value):");
            options.UseCVaR = false;
            options.RiskDistortion = 0.0;
            EvaluateModel(model, environment, 20);
            
            // Risk-averse evaluation (CVaR)
            Console.WriteLine("\nRisk-Averse Strategy (CVaR with alpha=0.25):");
            options.UseCVaR = true;
            options.CVaRAlpha = 0.25;
            options.RiskDistortion = 0.0;
            EvaluateModel(model, environment, 20);
            
            // Risk-distorted evaluation
            Console.WriteLine("\nRisk-Distorted Strategy (Distortion=0.5):");
            options.UseCVaR = false;
            options.RiskDistortion = 0.5;
            EvaluateModel(model, environment, 20);
            
            // 6. Visualize distribution prediction for a specific state
            Console.WriteLine("\nDistribution Prediction for Current Market State:");
            DisplayDistributionPrediction(model, environment.GetCurrentState());
            
            Console.WriteLine("\nQuantile Regression DQN Example completed successfully!");
        }
        
        /// <summary>
        /// Creates the options for the QR-DQN algorithm.
        /// </summary>
        /// <returns>The QR-DQN options.</returns>
        private static QRDQNOptions CreateQRDQNOptions()
        {
            return new QRDQNOptions
            {
                // State and action dimensions
                StateSize = 5,      // Market features (price, indicators, etc.)
                ActionSize = 3,     // Actions (buy, hold, sell)
                IsContinuous = false, // Discrete actions
                
                // QR-DQN specific parameters
                NumQuantiles = 50,  // Number of quantiles in the distribution
                HuberKappa = 1.0,   // Huber loss parameter
                UseCVaR = false,    // Initial setting (we'll adjust this later)
                CVaRAlpha = 0.25,   // Focus on worst 25% of outcomes
                RiskDistortion = 0.0, // Initial setting (we'll adjust this later)
                
                // Exploration parameters
                UseNoisyNetworks = true, // Use noisy networks for exploration
                InitialNoiseStd = 0.5,   // Initial noise standard deviation
                InitialExplorationRate = 1.0,   // Starting exploration rate (if not using noisy networks)
                FinalExplorationRate = 0.05,    // Minimum exploration rate
                // Note: Exploration decay is handled internally
                
                // Network architecture
                HiddenLayerSizes = new int[] { 64, 64 }, // Hidden layer sizes
                
                // Learning parameters
                Gamma = 0.95,          // Discount factor
                LearningRate = 0.001,  // Learning rate
                BatchSize = 32,        // Batch size for training
                ReplayBufferCapacity = 10000, // Replay buffer size
                EpochsPerUpdate = 4,   // How often to update the network
                TargetUpdateFrequency = 100, // How often to update the target network
                Tau = 1.0,             // Target network update rate (1.0 = hard update)
                
                // Double DQN and Prioritized Replay
                UseDoubleDQN = true,         // Use double DQN
                UsePrioritizedReplay = true, // Use prioritized experience replay
                PriorityAlpha = 0.6,         // Priority exponent
                PriorityBetaStart = 0.4      // Importance sampling correction
            };
        }
        
        /// <summary>
        /// Trains the model through interaction with the environment.
        /// </summary>
        /// <param name="model">The QR-DQN model.</param>
        /// <param name="environment">The stock environment.</param>
        /// <param name="episodes">The number of episodes to train for.</param>
        private static void TrainModel(QRDQNModel<double> model, QRDQNStockEnvironment environment, int episodes)
        {
            model.SetTrainingMode();
            
            double totalReward = 0;
            
            for (int episode = 0; episode < episodes; episode++)
            {
                // Reset environment for new episode
                environment.Reset();
                bool done = false;
                double episodeReward = 0;
                
                while (!done)
                {
                    // Get current state
                    var state = environment.GetCurrentState();
                    
                    // Select action using model
                    var action = model.SelectAction(state, true);
                    
                    // Take action in environment
                    var (nextState, reward, done_) = environment.Step(action);
                    done = done_;
                    
                    // Update model based on experience
                    model.Update(state, action, reward, nextState, done);
                    
                    // Accumulate reward
                    episodeReward += reward;
                }
                
                totalReward += episodeReward;
                
                // Print progress every few episodes
                if (episode % 10 == 0 || episode == episodes - 1)
                {
                    double averageReward = totalReward / (episode + 1);
                    Console.WriteLine($"Episode {episode + 1}/{episodes}, Latest Loss: {model.GetLoss():F4}, Average Reward: {averageReward:F2}");
                }
            }
            
            Console.WriteLine($"Training completed over {episodes} episodes with final average reward: {totalReward / episodes:F2}");
        }
        
        /// <summary>
        /// Evaluates the model on the environment.
        /// </summary>
        /// <param name="model">The QR-DQN model.</param>
        /// <param name="environment">The stock environment.</param>
        /// <param name="episodes">The number of episodes to evaluate for.</param>
        private static void EvaluateModel(QRDQNModel<double> model, QRDQNStockEnvironment environment, int episodes)
        {
            model.SetEvaluationMode();
            
            double totalReward = 0;
            int totalTrades = 0;
            int profitableTrades = 0;
            double maxDrawdown = 0;
            double peakValue = environment.InitialCapital;
            double currentDrawdown = 0;
            
            for (int episode = 0; episode < episodes; episode++)
            {
                // Reset environment for new episode
                environment.Reset();
                bool done = false;
                double episodeReward = 0;
                double episodeValue = environment.InitialCapital;
                
                while (!done)
                {
                    // Get current state
                    var state = environment.GetCurrentState();
                    
                    // Select action using model
                    var action = model.SelectAction(state, false);
                    
                    // Take action in environment
                    var (nextState, reward, done_) = environment.Step(action);
                    done = done_;
                    
                    // Update statistics
                    episodeReward += reward;
                    
                    // Track trades
                    int actionIndex = GetActionIndex(action);
                    if (actionIndex == 0) // Buy
                    {
                        totalTrades++;
                    }
                    else if (actionIndex == 2) // Sell
                    {
                        totalTrades++;
                        if (reward > 0)
                        {
                            profitableTrades++;
                        }
                    }
                    
                    // Track drawdown
                    double currentValue = environment.CurrentCapital;
                    if (currentValue > peakValue)
                    {
                        peakValue = currentValue;
                        currentDrawdown = 0;
                    }
                    else
                    {
                        currentDrawdown = (peakValue - currentValue) / peakValue;
                        if (currentDrawdown > maxDrawdown)
                        {
                            maxDrawdown = currentDrawdown;
                        }
                    }
                }
                
                totalReward += episodeReward;
                
                // Print episode result
                if (episode % 5 == 0 || episode == episodes - 1)
                {
                    Console.WriteLine($"Episode {episode + 1}, Reward: {episodeReward:F2}, Final Capital: ${environment.CurrentCapital:F2}");
                }
            }
            
            // Print overall statistics
            double averageReward = totalReward / episodes;
            double winRate = totalTrades > 0 ? (double)profitableTrades / totalTrades * 100 : 0;
            
            Console.WriteLine($"Evaluation Results:");
            Console.WriteLine($"  Average Reward: {averageReward:F2}");
            Console.WriteLine($"  Win Rate: {winRate:F2}%");
            Console.WriteLine($"  Maximum Drawdown: {maxDrawdown * 100:F2}%");
            Console.WriteLine($"  Risk-Adjusted Return: {(averageReward / (maxDrawdown > 0 ? maxDrawdown : 0.01)):F2}");
        }
        
        /// <summary>
        /// Displays the distribution prediction for a specific state.
        /// </summary>
        /// <param name="model">The QR-DQN model.</param>
        /// <param name="state">The state to evaluate.</param>
        private static void DisplayDistributionPrediction(QRDQNModel<double> model, Tensor<double> state)
        {
            // Since GetReturnDistribution is not available in the current implementation,
            // we'll display the model's action selection and confidence instead
            
            string[] actionNames = new[] { "Buy", "Hold", "Sell" };
            Console.WriteLine("Model Decision Analysis:");
            Console.WriteLine("------------------------------------");
            
            // Get the action the model would select
            var actionVector = model.SelectAction(state, isTraining: false);
            
            // Find which action was selected (highest value in the vector)
            int selectedAction = GetActionIndex(actionVector);
            
            Console.WriteLine($"Selected Action: {actionNames[selectedAction]}");
            Console.WriteLine($"Confidence: {actionVector[selectedAction]:P2}");
            
            // Display action values
            Console.WriteLine("\nAction Values:");
            for (int i = 0; i < actionNames.Length; i++)
            {
                Console.WriteLine($"  {actionNames[i],-10}: {actionVector[i]:F4}");
            }
            
            Console.WriteLine("\nNote: QR-DQN internally models return distributions,");
            Console.WriteLine("but the current implementation doesn't expose them directly.");
            Console.WriteLine("The model still uses distributional RL for robust decision-making.");
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
    public class QRDQNStockEnvironment
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
        /// Initializes a new instance of the <see cref="QRDQNStockEnvironment"/> class.
        /// </summary>
        /// <param name="initialPrice">The initial stock price.</param>
        public QRDQNStockEnvironment(double initialPrice)
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
                        
                        // Calculate profit/loss
                        double portfolioValue = CurrentCapital;
                        double pnl = (portfolioValue - InitialCapital) / InitialCapital;
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
            
            // Check if the episode is done
            _steps++;
            bool done = _steps >= _maxSteps;
            
            // If the episode is ending and we're still in a position, sell everything
            if (done && _inPosition)
            {
                _cash += _shares * _currentPrice;
                
                // Calculate final profit/loss
                double portfolioValue = CurrentCapital;
                double pnl = (portfolioValue - InitialCapital) / InitialCapital;
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