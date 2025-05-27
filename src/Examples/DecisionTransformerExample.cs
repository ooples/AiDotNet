using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Models;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.Memory;
using AiDotNet.Enums;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating the use of the Decision Transformer for stock market prediction and trading.
    /// </summary>
    public class DecisionTransformerExample
    {
        /// <summary>
        /// Runs the Decision Transformer example for stock market trading.
        /// </summary>
        public static void RunExample()
        {
            Console.WriteLine("Decision Transformer Example for Stock Trading");
            Console.WriteLine("=============================================");
            
            // 1. Configure the Decision Transformer
            var options = CreateDecisionTransformerOptions();
            
            // 2. Create a model
            var model = new DecisionTransformerModel<double>(options);
            Console.WriteLine("Model created successfully.");
            
            // 3. Generate or load some sample stock market data
            var data = GenerateSampleStockData(100);
            Console.WriteLine($"Generated {data.Count} days of sample stock data");
            
            // 4. Create training trajectories
            var trajectories = CreateTrajectories(data, 20);
            Console.WriteLine($"Created {trajectories.Count} training trajectories");
            
            // 5. Train the model offline
            Console.WriteLine("Training the model offline...");
            var result = model.TrainOffline(trajectories);
            Console.WriteLine("Training completed.");
            // TODO: Access appropriate property from result when available
            // Console.WriteLine($"Training completed with final loss: {result.Loss}");
            
            // 6. Evaluate the model
            Console.WriteLine("Evaluating the model...");
            model.SetEvaluationMode();
            EvaluateModel(model, data.Skip(80).Take(20).ToList());
            
            // 7. Save the model (optional)
            var modelPath = Path.Combine(Path.GetTempPath(), "decision_transformer_model.bin");
            SaveModel(model, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");
            
            Console.WriteLine("\nDecision Transformer Example completed successfully!");
        }
        
        /// <summary>
        /// Creates the options for the Decision Transformer.
        /// </summary>
        private static DecisionTransformerOptions<double> CreateDecisionTransformerOptions()
        {
            return new DecisionTransformerOptions<double>
            {
                // Basic settings
                StateSize = 10,                   // Features in the market state (price, volume, indicators, etc.)
                ActionSize = 3,                   // Number of actions (e.g., buy, sell, hold)
                IsContinuous = false,             // Discrete actions for simplicity
                
                // Decision Transformer specific settings
                ContextLength = 20,               // Consider the last 20 days for decision making
                NumTransformerLayers = 3,         // Number of transformer layers
                NumHeads = 4,                     // Number of attention heads
                EmbeddingDim = 64,                // Embedding dimension
                ReturnConditioned = true,         // Condition on target returns
                OfflineTraining = true,           // Train on historical data
                TransformerLearningRate = 1e-4,   // Learning rate for transformer
                DropoutRate = 0.1,                // Dropout rate for regularization
                PositionalEncodingType = PositionalEncodingType.Sinusoidal, // Type of positional encoding
                
                // General reinforcement learning settings
                Gamma = 0.99,                     // Discount factor
                BatchSize = 8,                    // Batch size for training
                MaxBufferSize = 10000,            // Maximum buffer size
                UpdateFrequency = 10,             // Frequency of model updates
                MaxTrajectoryLength = 50          // Maximum trajectory length
            };
        }
        
        /// <summary>
        /// Generates sample stock market data.
        /// </summary>
        /// <param name="days">The number of days to generate.</param>
        /// <returns>A list of stock market states.</returns>
        private static List<(Tensor<double> state, double price)> GenerateSampleStockData(int days)
        {
            var data = new List<(Tensor<double> state, double price)>();
            var random = new Random(42);  // Fixed seed for reproducibility
            
            double price = 100.0;  // Starting price
            double trend = 0.0;    // Trend component
            double volatility = 1.0; // Volatility
            
            for (int day = 0; day < days; day++)
            {
                // Simulate price movement
                if (day % 20 == 0)
                {
                    // Occasional trend changes
                    trend = (random.NextDouble() - 0.5) * 0.3;
                }
                
                double dailyReturn = trend + (random.NextDouble() - 0.5) * volatility;
                price *= (1 + dailyReturn);
                
                // Create features vector for this day
                var features = new Vector<double>(10);
                
                // Feature 1: Normalized price
                features[0] = price / 100.0;
                
                // Feature 2: Daily return
                features[1] = dailyReturn;
                
                // Feature 3: 5-day momentum (just a placeholder in this example)
                features[2] = day >= 5 ? (price / data[Math.Max(0, day - 5)].price) - 1.0 : 0.0;
                
                // Feature 4: Volatility estimate
                features[3] = volatility;
                
                // Feature 5: Day of week (normalized)
                features[4] = (day % 5) / 4.0;
                
                // Features 6-10: Random technical indicators
                for (int i = 5; i < 10; i++)
                {
                    features[i] = (random.NextDouble() - 0.5) * 2.0;  // Random values between -1 and 1
                }
                
                // Create a state tensor from the features
                var stateTensor = Tensor<double>.FromVector(features);
                
                // Add to dataset
                data.Add((stateTensor, price));
            }
            
            return data;
        }
        
        /// <summary>
        /// Creates training trajectories from stock market data.
        /// </summary>
        /// <param name="data">The stock market data.</param>
        /// <param name="trajectoryCount">The number of trajectories to create.</param>
        /// <returns>A list of trajectory batches.</returns>
        private static List<TrajectoryBatch<Tensor<double>, Vector<double>, double>> CreateTrajectories(
            List<(Tensor<double> state, double price)> data, int trajectoryCount)
        {
            var trajectories = new List<TrajectoryBatch<Tensor<double>, Vector<double>, double>>();
            var random = new Random(42);  // Fixed seed for reproducibility
            
            for (int t = 0; t < trajectoryCount; t++)
            {
                // For each trajectory, select a random starting point
                int startIdx = random.Next(0, data.Count - 20);  // Ensure at least 20 days of data
                int length = random.Next(10, 20);  // Random length between 10 and 20
                
                // Arrays to store the trajectory data
                var states = new Tensor<double>[length];
                var actions = new Vector<double>[length];
                var rewards = new double[length];
                var nextStates = new Tensor<double>[length];
                var dones = new bool[length];
                
                double cumulativeReturn = 0.0;
                int position = 0;  // 0: no position, 1: long position
                
                for (int i = 0; i < length; i++)
                {
                    int dayIdx = startIdx + i;
                    
                    // Current state
                    states[i] = data[dayIdx].state;
                    
                    // Simulate a simple trading strategy for demonstration
                    Vector<double> action;
                    
                    if (i > 0)
                    {
                        // Simple momentum strategy: buy if price increased, sell if decreased
                        double priceChange = (data[dayIdx].price / data[dayIdx - 1].price) - 1.0;
                        
                        if (priceChange > 0.01 && position == 0)
                        {
                            // Buy signal
                            action = new Vector<double>(3) { [0] = 0, [1] = 1, [2] = 0 };  // Buy
                            position = 1;
                        }
                        else if (priceChange < -0.01 && position == 1)
                        {
                            // Sell signal
                            action = new Vector<double>(3) { [0] = 0, [1] = 0, [2] = 1 };  // Sell
                            position = 0;
                        }
                        else
                        {
                            // Hold
                            action = new Vector<double>(3) { [0] = 1, [1] = 0, [2] = 0 };  // Hold
                        }
                    }
                    else
                    {
                        // First day, just hold
                        action = new Vector<double>(3) { [0] = 1, [1] = 0, [2] = 0 };  // Hold
                    }
                    
                    actions[i] = action;
                    
                    // Compute reward based on action and price change
                    double nextDayReturn = 0.0;
                    if (dayIdx < data.Count - 1)
                    {
                        nextDayReturn = (data[dayIdx + 1].price / data[dayIdx].price) - 1.0;
                    }
                    
                    double reward = 0.0;
                    
                    if (action[1] > 0)  // Buy
                    {
                        reward = 0.0;  // Cost of transaction
                    }
                    else if (action[2] > 0)  // Sell
                    {
                        reward = position * nextDayReturn;  // Realized gains/losses
                    }
                    else  // Hold
                    {
                        reward = position * nextDayReturn;  // Unrealized gains/losses
                    }
                    
                    rewards[i] = reward;
                    cumulativeReturn += reward;
                    
                    // Next state
                    nextStates[i] = dayIdx < data.Count - 1 ? data[dayIdx + 1].state : data[dayIdx].state;
                    
                    // Done flag (only true for the last step)
                    dones[i] = i == length - 1;
                }
                
                // Create and add the trajectory batch
                var batch = new TrajectoryBatch<Tensor<double>, Vector<double>, double>(
                    states, actions, rewards, nextStates, dones);
                trajectories.Add(batch);
            }
            
            return trajectories;
        }
        
        /// <summary>
        /// Evaluates the model on test data.
        /// </summary>
        /// <param name="model">The Decision Transformer model.</param>
        /// <param name="testData">The test data.</param>
        private static void EvaluateModel(
            DecisionTransformerModel<double> model, List<(Tensor<double> state, double price)> testData)
        {
            double initialCapital = 10000.0;
            double capital = initialCapital;
            int position = 0;  // 0: no position, 1: long position
            double sharePrice = testData[0].price;
            double shares = 0;
            
            Console.WriteLine("\nTrading Simulation:");
            Console.WriteLine($"Starting capital: ${initialCapital:F2}");
            
            for (int day = 0; day < testData.Count; day++)
            {
                // Get state and ask model for action
                var state = testData[day].state;
                var actionVector = model.SelectAction(state);
                
                // Convert to action type
                int actionType = 0; // 0: hold, 1: buy, 2: sell
                double maxProb = actionVector[0];
                
                for (int i = 1; i < actionVector.Length; i++)
                {
                    if (actionVector[i] > maxProb)
                    {
                        maxProb = actionVector[i];
                        actionType = i;
                    }
                }
                
                string actionName = actionType == 0 ? "HOLD" : (actionType == 1 ? "BUY" : "SELL");
                
                // Execute action
                sharePrice = testData[day].price;
                
                if (actionType == 1 && position == 0)  // Buy
                {
                    shares = capital / sharePrice;
                    capital = 0;
                    position = 1;
                    Console.WriteLine($"Day {day}: {actionName} {shares:F2} shares at ${sharePrice:F2}");
                }
                else if (actionType == 2 && position == 1)  // Sell
                {
                    capital = shares * sharePrice;
                    shares = 0;
                    position = 0;
                    Console.WriteLine($"Day {day}: {actionName} for ${capital:F2}");
                }
                else
                {
                    // Hold - no action needed
                }
            }
            
            // Final portfolio value
            double finalValue = capital + (shares * sharePrice);
            double totalReturn = (finalValue - initialCapital) / initialCapital * 100;
            
            Console.WriteLine($"\nFinal Portfolio Value: ${finalValue:F2}");
            Console.WriteLine($"Total Return: {totalReturn:F2}%");
            Console.WriteLine($"Position: {(position == 1 ? $"{shares:F2} shares" : "Cash")}");
        }
        
        /// <summary>
        /// Saves the model to a file.
        /// </summary>
        /// <param name="model">The model to save.</param>
        /// <param name="filePath">The file path to save the model to.</param>
        private static void SaveModel(DecisionTransformerModel<double> model, string filePath)
        {
            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                model.Save(fileStream);
            }
        }
        
        /// <summary>
        /// Loads a model from a file.
        /// </summary>
        /// <param name="options">The options for the model.</param>
        /// <param name="filePath">The file path to load the model from.</param>
        /// <returns>The loaded model.</returns>
        private static DecisionTransformerModel<double> LoadModel(DecisionTransformerOptions<double> options, string filePath)
        {
            var model = new DecisionTransformerModel<double>(options);
            
            using (var fileStream = new FileStream(filePath, FileMode.Open))
            {
                model.Load(fileStream);
            }
            
            return model;
        }
    }
}