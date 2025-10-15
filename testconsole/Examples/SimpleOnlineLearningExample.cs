using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Simple example demonstrating online learning in AiDotNet
    /// </summary>
    public class SimpleOnlineLearningExample
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== AiDotNet Online Learning Example ===\n");
            
            // Example 1: Online Classification
            Console.WriteLine("1. Online Binary Classification with Perceptron");
            Console.WriteLine("----------------------------------------------");
            RunPerceptronExample();
            
            Console.WriteLine("\n2. Online Regression with SGD");
            Console.WriteLine("-----------------------------");
            RunSGDRegressionExample();
            
            Console.WriteLine("\n3. Online Clustering with K-Means");
            Console.WriteLine("---------------------------------");
            RunKMeansExample();
            
            Console.WriteLine("\n=== Example Complete ===");
        }
        
        private static void RunPerceptronExample()
        {
            // Create an online perceptron for 2D classification
            var perceptron = new OnlinePerceptron<double>(2);
            
            // Generate some training data
            var random = new Random(42);
            var trainingData = new List<(Vector<double> input, double label)>();
            
            // Generate 100 linearly separable points
            for (int i = 0; i < 100; i++)
            {
                var x = random.NextDouble() * 10 - 5;
                var y = random.NextDouble() * 10 - 5;
                
                // Simple linear boundary: points above y = x are positive
                var label = y > x ? 1.0 : -1.0;
                
                trainingData.Add((new Vector<double>(new[] { x, y }), label));
            }
            
            // Train the model online (one sample at a time)
            Console.WriteLine("Training perceptron on 100 samples...");
            foreach (var (input, label) in trainingData)
            {
                perceptron.PartialFit(input, label);
            }
            
            // Test the model
            Console.WriteLine("\nTesting on new points:");
            var testPoints = new[]
            {
                new Vector<double>(new[] { 2.0, 3.0 }),   // Above line (positive)
                new Vector<double>(new[] { 3.0, 2.0 }),   // Below line (negative)
                new Vector<double>(new[] { 0.0, 0.0 }),   // On line
                new Vector<double>(new[] { -1.0, 1.0 }),  // Above line (positive)
                new Vector<double>(new[] { 1.0, -1.0 })   // Below line (negative)
            };
            
            foreach (var point in testPoints)
            {
                var prediction = perceptron.Predict(point);
                var predictedClass = prediction > 0 ? "Positive" : "Negative";
                Console.WriteLine($"  Point ({point[0]:F1}, {point[1]:F1}) -> {predictedClass}");
            }
        }
        
        private static void RunSGDRegressionExample()
        {
            // Create an online SGD regressor with options
            var options = new AdaptiveOnlineModelOptions<double>
            {
                InitialLearningRate = 0.01,
                RegularizationParameter = 0.001 // L2 regularization
            };
            
            var sgdRegressor = new OnlineSGDRegressor<double>(1, FitnessCalculatorType.MeanSquaredError, options);
            
            // Generate noisy linear data: y = 2x + 1 + noise
            var random = new Random(42);
            Console.WriteLine("Training SGD regressor on noisy linear data (y = 2x + 1)...");
            
            // Train in mini-batches
            for (int batch = 0; batch < 20; batch++)
            {
                var batchInputs = new List<Vector<double>>();
                var batchTargets = new List<double>();
                
                // Create mini-batch of 10 samples
                for (int i = 0; i < 10; i++)
                {
                    var x = random.NextDouble() * 10;
                    var noise = (random.NextDouble() - 0.5) * 0.5;
                    var y = 2 * x + 1 + noise;
                    
                    batchInputs.Add(new Vector<double>(new[] { x }));
                    batchTargets.Add(y);
                }
                
                // Update model with batch
                sgdRegressor.PartialFitBatch(batchInputs.ToArray(), batchTargets.ToArray());
                
                if ((batch + 1) % 5 == 0)
                {
                    Console.WriteLine($"  Completed batch {batch + 1}/20");
                }
            }
            
            // Test the model
            Console.WriteLine("\nTesting regression model:");
            Console.WriteLine("X    True Y    Predicted Y    Error");
            Console.WriteLine("------------------------------------");
            
            for (double x = 0; x <= 10; x += 2)
            {
                var trueY = 2 * x + 1;
                var prediction = sgdRegressor.Predict(new Vector<double>(new[] { x }));
                var error = Math.Abs(prediction - trueY);
                
                Console.WriteLine($"{x:F0}    {trueY:F1}        {prediction:F2}         {error:F2}");
            }
        }
        
        private static void RunKMeansExample()
        {
            // Create online K-means with 3 clusters
            var kmeans = new OnlineKMeans<double>(2, k: 3);
            
            // Generate data from 3 different clusters
            var random = new Random(42);
            var clusterCenters = new[]
            {
                new[] { 2.0, 2.0 },
                new[] { -2.0, 2.0 },
                new[] { 0.0, -3.0 }
            };
            
            Console.WriteLine("Streaming data points from 3 clusters...");
            
            // Stream data points one by one
            var pointsPerCluster = 30;
            var allPoints = new List<Vector<double>>();
            
            for (int i = 0; i < pointsPerCluster * 3; i++)
            {
                // Select cluster randomly
                var cluster = random.Next(3);
                
                // Generate point near that cluster center
                var x = clusterCenters[cluster][0] + (random.NextDouble() - 0.5) * 1.5;
                var y = clusterCenters[cluster][1] + (random.NextDouble() - 0.5) * 1.5;
                
                var point = new Vector<double>(new[] { x, y });
                allPoints.Add(point);
                
                // Update clustering model
                kmeans.PartialFit(point, 0); // Target is ignored for clustering
                
                if ((i + 1) % 30 == 0)
                {
                    Console.WriteLine($"  Processed {i + 1} points");
                }
            }
            
            // Show final cluster centers
            Console.WriteLine("\nLearned cluster centers:");
            var metadata = kmeans.GetModelMetaData();
            var centroids = metadata.AdditionalInfo["Centroids"] as Vector<double>[];
            
            for (int i = 0; i < centroids!.Length; i++)
            {
                Console.WriteLine($"  Cluster {i}: ({centroids[i][0]:F2}, {centroids[i][1]:F2})");
            }
            
            // Assign some test points
            Console.WriteLine("\nClustering new points:");
            var testPoints = new[]
            {
                new Vector<double>(new[] { 2.5, 2.5 }),
                new Vector<double>(new[] { -2.5, 2.5 }),
                new Vector<double>(new[] { 0.0, -3.5 }),
                new Vector<double>(new[] { 0.0, 0.0 })
            };
            
            foreach (var point in testPoints)
            {
                var cluster = (int)kmeans.Predict(point);
                Console.WriteLine($"  Point ({point[0]:F1}, {point[1]:F1}) -> Cluster {cluster}");
            }
        }
    }
}