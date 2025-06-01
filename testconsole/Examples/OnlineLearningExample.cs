using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;
using AiDotNet.OnlineLearning;

namespace TestConsoleApp.Examples
{
    public static class OnlineLearningExample
    {
        public static void Run()
        {
            Console.WriteLine("=== Online Learning Example ===\n");
            
            // Example 1: Simple Online Perceptron
            SimplePerceptronExample();
            
            Console.WriteLine("\n" + new string('-', 50) + "\n");
            
            // Example 2: Passive-Aggressive Regression
            PassiveAggressiveExample();
            
            Console.WriteLine("\n" + new string('-', 50) + "\n");
            
            // Example 3: Online SGD with Drift Detection
            OnlineSGDWithDriftExample();
            
            Console.WriteLine("\n" + new string('-', 50) + "\n");
            
            // Example 4: Streaming Data Simulation
            StreamingDataExample();
            
            Console.WriteLine("\n=== Online Learning Example Completed ===");
        }
        
        private static void SimplePerceptronExample()
        {
            Console.WriteLine("1. Simple Online Perceptron for Binary Classification");
            Console.WriteLine("-----------------------------------------------------");
            
            // Generate linearly separable data
            var random = new Random(42);
            var dataSize = 200;
            
            Console.WriteLine($"Generating {dataSize} linearly separable data points...");
            
            // Create perceptron with 2 input features
            var perceptron = new OnlinePerceptron<double>(
                inputDimension: 2,
                options: new OnlineModelOptions<double>
                {
                    InitialLearningRate = 0.1,
                    UseAdaptiveLearningRate = true,
                    LearningRateDecay = 0.001
                }
            );
            
            // Train incrementally
            Console.WriteLine("\nTraining perceptron incrementally...");
            int correctPredictions = 0;
            
            for (int i = 0; i < dataSize; i++)
            {
                // Generate a point
                double x1 = random.NextDouble() * 10 - 5;
                double x2 = random.NextDouble() * 10 - 5;
                
                // Label based on which side of line x1 + x2 = 0
                double label = (x1 + x2 > 0) ? 1.0 : 0.0;
                
                var input = new Vector<double>(new[] { x1, x2 });
                
                // Predict before update
                var prediction = perceptron.Predict(input);
                if (Math.Abs(prediction - label) < 0.5)
                {
                    correctPredictions++;
                }
                
                // Update model
                perceptron.PartialFit(input, label);
                
                if ((i + 1) % 50 == 0)
                {
                    double accuracy = (double)correctPredictions / (i + 1) * 100;
                    Console.WriteLine($"  After {i + 1} samples: Accuracy = {accuracy:F2}%");
                }
            }
            
            // Test on new data
            Console.WriteLine("\nTesting on 50 new samples...");
            int testCorrect = 0;
            for (int i = 0; i < 50; i++)
            {
                double x1 = random.NextDouble() * 10 - 5;
                double x2 = random.NextDouble() * 10 - 5;
                double label = (x1 + x2 > 0) ? 1.0 : 0.0;
                
                var input = new Vector<double>(new[] { x1, x2 });
                var prediction = perceptron.Predict(input);
                
                if (Math.Abs(prediction - label) < 0.5)
                {
                    testCorrect++;
                }
            }
            
            Console.WriteLine($"Test accuracy: {testCorrect / 50.0 * 100:F2}%");
            Console.WriteLine($"Total samples seen: {perceptron.SamplesSeen}");
        }
        
        private static void PassiveAggressiveExample()
        {
            Console.WriteLine("2. Passive-Aggressive Regression");
            Console.WriteLine("---------------------------------");
            
            // Generate noisy linear data
            var random = new Random(42);
            var dataSize = 300;
            
            Console.WriteLine($"Generating {dataSize} noisy linear regression data points...");
            
            // Create PA regressor
            var paOptions = new OnlineModelOptions<double>
            {
                AggressivenessParameter = 1.0,
                Epsilon = 0.1
            };
            var paRegressor = new PassiveAggressiveRegressor<double>(
                inputDimension: 3,
                options: paOptions
            );
            
            Console.WriteLine("\nTraining PA regressor incrementally...");
            
            int windowSize = 50;
            var recentErrors = new Queue<double>(windowSize);
            
            for (int i = 0; i < dataSize; i++)
            {
                // Generate features
                var x1 = random.NextDouble() * 10;
                var x2 = random.NextDouble() * 10;
                var x3 = random.NextDouble() * 10;
                
                // True function: y = 2*x1 - 3*x2 + x3 + noise
                var noise = random.NextDouble() * 2 - 1; // [-1, 1]
                var y = 2 * x1 - 3 * x2 + x3 + noise;
                
                // Add occasional outliers
                if (random.NextDouble() < 0.05) // 5% outliers
                {
                    y += random.NextDouble() * 20 - 10; // Large noise
                }
                
                var input = new Vector<double>(new[] { x1, x2, x3 });
                
                // Predict before update
                var prediction = paRegressor.Predict(input);
                var error = Math.Abs(prediction - y);
                
                recentErrors.Enqueue(error);
                if (recentErrors.Count > windowSize)
                {
                    recentErrors.Dequeue();
                }
                
                // Update model
                paRegressor.PartialFit(input, y);
                
                if ((i + 1) % 100 == 0)
                {
                    var avgError = recentErrors.Average();
                    Console.WriteLine($"  After {i + 1} samples: Avg error (last {windowSize}) = {avgError:F4}");
                }
            }
            
            // Test final performance
            Console.WriteLine("\nTesting on 50 new samples...");
            double testError = 0;
            for (int i = 0; i < 50; i++)
            {
                var x1 = random.NextDouble() * 10;
                var x2 = random.NextDouble() * 10;
                var x3 = random.NextDouble() * 10;
                var y = 2 * x1 - 3 * x2 + x3;
                
                var input = new Vector<double>(new[] { x1, x2, x3 });
                var prediction = paRegressor.Predict(input);
                testError += Math.Abs(prediction - y);
            }
            
            Console.WriteLine($"Average test error: {testError / 50:F4}");
            Console.WriteLine($"Total samples seen: {paRegressor.SamplesSeen}");
        }
        
        private static void OnlineSGDWithDriftExample()
        {
            Console.WriteLine("3. Online SGD with Concept Drift Detection");
            Console.WriteLine("------------------------------------------");
            
            var random = new Random(42);
            var dataSize = 500;
            
            Console.WriteLine("Simulating data stream with concept drift...");
            
            // Create adaptive SGD regressor
            var sgdOptions = new AdaptiveOnlineModelOptions<double>
            {
                InitialLearningRate = 0.01,
                UseAdaptiveLearningRate = true,
                UseMomentum = true,
                MomentumFactor = 0.9,
                DriftSensitivity = 0.5
            };
            var sgdRegressor = new OnlineSGDRegressor<double>(
                inputDimension: 2,
                lossType: FitnessCalculatorType.MeanSquaredError,
                options: sgdOptions,
                driftMethod: DriftDetectionMethod.ADWIN
            );
            
            Console.WriteLine("\nTraining with changing data patterns...");
            
            for (int i = 0; i < dataSize; i++)
            {
                // Generate features
                var x1 = random.NextDouble() * 10;
                var x2 = random.NextDouble() * 10;
                
                // Change the true function halfway through (concept drift)
                double y;
                if (i < dataSize / 2)
                {
                    // First concept: y = x1 + x2
                    y = x1 + x2 + (random.NextDouble() * 0.5 - 0.25);
                }
                else
                {
                    // Second concept: y = 2*x1 - x2 (drift occurs)
                    y = 2 * x1 - x2 + (random.NextDouble() * 0.5 - 0.25);
                }
                
                var input = new Vector<double>(new[] { x1, x2 });
                
                // Update model
                sgdRegressor.PartialFit(input, y);
                
                // Check for drift
                if (sgdRegressor.DriftDetected)
                {
                    Console.WriteLine($"  *** Drift detected at sample {i + 1}! Drift level: {sgdRegressor.DriftLevel:F4}");
                }
                
                if ((i + 1) % 100 == 0)
                {
                    Console.WriteLine($"  After {i + 1} samples: Learning rate = {sgdRegressor.LearningRate:F6}");
                }
            }
            
            Console.WriteLine($"\nTotal samples seen: {sgdRegressor.SamplesSeen}");
            Console.WriteLine($"Final drift level: {sgdRegressor.DriftLevel:F4}");
        }
        
        private static void StreamingDataExample()
        {
            Console.WriteLine("4. Simulating Streaming Data Processing");
            Console.WriteLine("---------------------------------------");
            
            Console.WriteLine("Simulating a data stream that arrives in batches...\n");
            
            // Create an online model
            var model = new OnlinePerceptron<double>(
                inputDimension: 4,
                options: new OnlineModelOptions<double>
                {
                    InitialLearningRate = 0.1,
                    MiniBatchSize = 10
                }
            );
            
            var random = new Random(42);
            int totalBatches = 10;
            int samplesPerBatch = 20;
            
            // Simulate streaming batches
            for (int batch = 0; batch < totalBatches; batch++)
            {
                Console.WriteLine($"Processing batch {batch + 1}/{totalBatches}...");
                
                // Generate batch data
                var batchInputs = new Vector<double>[samplesPerBatch];
                var batchOutputs = new double[samplesPerBatch];
                
                for (int i = 0; i < samplesPerBatch; i++)
                {
                    var features = new double[4];
                    for (int j = 0; j < 4; j++)
                    {
                        features[j] = random.NextDouble() * 10 - 5;
                    }
                    
                    // Complex decision boundary
                    var label = (features[0] + features[1] > features[2] + features[3]) ? 1.0 : 0.0;
                    
                    batchInputs[i] = new Vector<double>(features);
                    batchOutputs[i] = label;
                }
                
                // Process batch
                model.PartialFitBatch(batchInputs, batchOutputs);
                
                // Evaluate on batch
                int correct = 0;
                for (int i = 0; i < samplesPerBatch; i++)
                {
                    var prediction = model.Predict(batchInputs[i]);
                    if (Math.Abs(prediction - batchOutputs[i]) < 0.5)
                    {
                        correct++;
                    }
                }
                
                double accuracy = (double)correct / samplesPerBatch * 100;
                Console.WriteLine($"  Batch accuracy: {accuracy:F2}%");
                Console.WriteLine($"  Total samples processed: {model.SamplesSeen}");
                
                // Simulate delay between batches
                System.Threading.Thread.Sleep(100);
            }
            
            Console.WriteLine($"\nStream processing completed. Total samples: {model.SamplesSeen}");
        }
    }
}