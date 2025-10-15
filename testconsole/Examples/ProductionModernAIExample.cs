using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using AiDotNet.AutoML;
using AiDotNet.Deployment;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Pipeline;
using AiDotNet.Regression;
using AiDotNet.ProductionMonitoring;
using AiDotNet.Interpretability;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Production-ready example demonstrating modern AI features in AiDotNet
    /// </summary>
    public static class ProductionModernAIExample
    {
        /// <summary>
        /// Demonstrates using AutoML to automatically find the best model
        /// </summary>
        public static async Task AutoMLExampleAsync()
        {
            Console.WriteLine("=== AutoML Example ===");
            
            // Create realistic sample data
            var (features, targets) = GenerateRegressionData(100, 5);
            
            // Create an AutoML model
            var autoML = new BayesianOptimizationAutoML<double>();
            
            // Configure search space with realistic parameters
            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["learning_rate"] = new ContinuousParameterRange(0.001, 0.1),
                ["hidden_units"] = new IntegerParameterRange(10, 100),
                ["activation"] = new CategoricalParameterRange(new[] { "ReLU", "Tanh", "Sigmoid" }),
                ["dropout_rate"] = new ContinuousParameterRange(0.0, 0.5),
                ["batch_size"] = new IntegerParameterRange(16, 64),
                ["epochs"] = new IntegerParameterRange(50, 200)
            };
            
            autoML.SetSearchSpace(searchSpace);
            
            // Configure candidate models
            autoML.SetCandidateModels(new List<ModelType> 
            { 
                ModelType.LinearRegression, 
                ModelType.NeuralNetwork, 
                ModelType.RandomForest,
                ModelType.GradientBoosting
            });
            
            // Set optimization metric
            autoML.SetOptimizationMetric(MetricType.RMSE, maximize: false);
            
            // Enable early stopping
            autoML.EnableEarlyStopping(patience: 10, minDelta: 0.001);
            
            // Split data for validation
            var splitStep = new DataSplittingStep(0.8, 0.1, 0.1, randomSeed: 42);
            await splitStep.FitAsync(features, targets);
            var (trainData, trainTargets) = splitStep.GetTrainData(features, targets);
            var (valData, valTargets) = splitStep.GetValidationData(features, targets);
            
            // Run AutoML search
            var bestModel = await autoML.SearchAsync(
                new Matrix<double>(trainData),
                new Vector<double>(trainTargets),
                new Matrix<double>(valData),
                new Vector<double>(valTargets),
                timeLimit: TimeSpan.FromMinutes(5),
                CancellationToken.None);
            
            // Wrap the best model with monitoring capabilities
            var monitoredModel = new MonitoredModelWrapper<double, Matrix<double>, Vector<double>>(bestModel);
            
            // Display results
            Console.WriteLine($"AutoML Status: {autoML.Status}");
            Console.WriteLine($"Best Score (RMSE): {autoML.BestScore:F4}");
            Console.WriteLine($"Best Model Type: {bestModel.GetModelMetaData().ModelType}");
            
            // Use the monitored model for predictions
            var testPrediction = monitoredModel.Predict(new Matrix<double>(trainData.Take(1).ToArray()));
            Console.WriteLine($"\nMonitored prediction made. Health score: {monitoredModel.GetHealthScore():F2}");
            
            // Get feature importance
            var featureImportance = await autoML.GetFeatureImportanceAsync();
            Console.WriteLine("\nFeature Importance:");
            foreach (var (feature, importance) in featureImportance.OrderByDescending(x => x.Value))
            {
                Console.WriteLine($"  Feature {feature}: {importance:F4}");
            }
        }
        
        /// <summary>
        /// Demonstrates multimodal AI combining different data types
        /// </summary>
        public static async Task MultimodalExampleAsync()
        {
            Console.WriteLine("\n=== Multimodal AI Example ===");
            
            // Create a multimodal model
            var multimodalModel = new LateFusionMultimodal(fusedSize: 256);
            
            // The model already has built-in encoders for common modalities
            
            // Prepare multimodal data
            var modalityData = new Dictionary<string, object>
            {
                ["text"] = new[] { "High quality product", "Average item", "Excellent value" },
                ["numerical"] = new double[,] { { 4.5, 100 }, { 3.0, 50 }, { 5.0, 120 } }
            };
            
            // Process multimodal input
            var embeddings = multimodalModel.ProcessMultimodal(modalityData);
            
            Console.WriteLine($"Multimodal embedding dimension: {embeddings.Length}");
            Console.WriteLine($"Fusion strategy: {multimodalModel.FusionStrategy}");
            Console.WriteLine($"Supported modalities: {string.Join(", ", multimodalModel.SupportedModalities)}");
        }
        
        /// <summary>
        /// Demonstrates model interpretability and explainability
        /// </summary>
        public static async Task InterpretabilityExampleAsync()
        {
            Console.WriteLine("\n=== Model Interpretability Example ===");
            
            // Create sample data
            var (features, targets) = GenerateRegressionData(100, 5);
            
            // Train a simple model
            var model = new SimpleRegression();
            var trainData = features.Select(f => new Vector<double>(f)).ToArray();
            
            for (int i = 0; i < trainData.Length; i++)
            {
                model.Train(trainData[i], targets[i]);
            }
            
            // TODO: Update this example to use a model that implements IFullModel
            // which now includes interpretability features by default
            Console.WriteLine("\nNote: Interpretability features are now part of IFullModel interface.");
            Console.WriteLine("Models implementing IFullModel can provide explanations directly.");
        }
        
        /// <summary>
        /// Demonstrates production monitoring and drift detection
        /// </summary>
        public static async Task ProductionMonitoringExampleAsync()
        {
            Console.WriteLine("\n=== Production Monitoring Example ===");
            
            // Create sample model and data
            var (trainFeatures, trainTargets) = GenerateRegressionData(100, 5);
            var model = new SimpleRegression();
            
            // Train model
            for (int i = 0; i < trainFeatures.Length; i++)
            {
                model.Train(new Vector<double>(trainFeatures[i]), trainTargets[i]);
            }
            
            // Create production monitor
            var monitor = new DefaultProductionMonitor<double>();
            
            // Simulate production data with drift
            var (prodFeatures, prodTargets) = GenerateRegressionData(50, 5, drift: true);
            
            // Monitor predictions
            for (int i = 0; i < prodFeatures.Length; i++)
            {
                var input = new Vector<double>(prodFeatures[i]);
                var prediction = model.Predict(input);
                
                // Monitor the prediction - DefaultProductionMonitor doesn't track individual predictions
                // Instead, it monitors drift in batches
                await monitor.MonitorPredictionsAsync(new Tensor<double>(new[] { prediction }));
            }
            
            // Get monitoring results
            var recentAlerts = monitor.GetRecentAlerts();
            if (recentAlerts.Any())
            {
                Console.WriteLine("\nAlerts detected:");
                foreach (var alert in recentAlerts)
                {
                    Console.WriteLine($"  - {alert.Message} (Severity: {alert.Severity})");
                }
            }
            
            // Check if retraining is needed
            if (monitor.GetRetrainingRecommendation())
            {
                Console.WriteLine("\nWarning: Model retraining recommended!");
            }
            
            // Get model health score
            var healthScore = monitor.GetHealthScore();
            Console.WriteLine($"\nModel Health Score: {healthScore:F2}");
        }
        
        /// <summary>
        /// Demonstrates deployment optimization for different platforms
        /// </summary>
        public static async Task DeploymentOptimizationExampleAsync()
        {
            Console.WriteLine("\n=== Deployment Optimization Example ===");
            
            // Create and train a neural network
            var (features, targets) = GenerateRegressionData(100, 10);
            
            var neuralNet = new NeuralNetwork();
            neuralNet.AddLayer(LayerType.Dense, 10, ActivationFunction.ReLU);
            neuralNet.AddLayer(LayerType.Dense, 20, ActivationFunction.ReLU);
            neuralNet.AddLayer(LayerType.Dense, 10, ActivationFunction.ReLU);
            neuralNet.AddLayer(LayerType.Dense, 1, ActivationFunction.Linear);
            
            // Compile and train briefly
            neuralNet.Compile(LossFunction.MeanSquaredError, OptimizerType.Adam);
            
            // Create cloud optimizer
            var cloudOptimizer = new CloudOptimizer<double>();
            var cloudOptimized = await cloudOptimizer.OptimizeForCloudAsync(
                neuralNet,
                new CloudOptimizationOptions
                {
                    Platform = CloudPlatform.AWS,
                    EnableAutoScaling = true,
                    TargetLatencyMs = 100
                });
            
            Console.WriteLine("Cloud Optimization Results:");
            Console.WriteLine($"  Original Size: {GetModelSize(neuralNet):F2} MB");
            Console.WriteLine($"  Optimized Size: {GetModelSize(cloudOptimized):F2} MB");
            Console.WriteLine($"  Inference Speed: {cloudOptimizer.EstimatedLatencyMs} ms");
            
            // Create edge optimizer
            var edgeOptimizer = new EdgeOptimizer<double>();
            var edgeOptimized = await edgeOptimizer.OptimizeForEdgeAsync(
                neuralNet,
                new EdgeOptimizationOptions
                {
                    DeviceType = "Mobile",
                    MemoryLimitMB = 50,
                    TargetLatencyMs = 10,
                    Quantization = QuantizationType.Int8
                });
            
            Console.WriteLine("\nEdge Optimization Results:");
            Console.WriteLine($"  Model Size: {GetModelSize(edgeOptimized):F2} MB");
            Console.WriteLine($"  Memory Usage: {edgeOptimizer.EstimatedMemoryMB} MB");
            Console.WriteLine($"  Battery Impact: {edgeOptimizer.EstimatedBatteryImpact}");
        }
        
        /// <summary>
        /// Demonstrates advanced pipeline with custom steps
        /// </summary>
        public static async Task AdvancedPipelineExampleAsync()
        {
            Console.WriteLine("\n=== Advanced Pipeline Example ===");
            
            // Create sample data
            var (features, targets) = GenerateRegressionData(200, 5);
            
            // Create advanced pipeline
            var pipeline = new List<IPipelineStep>
            {
                // Data cleaning
                new DataCleaningStep(new DataCleaningConfig
                {
                    HandleOutliers = true,
                    OutlierMethod = OutlierDetectionMethod.IQR,
                    HandleMissingValues = true,
                    ImputationStrategy = ImputationStrategy.Mean
                }),
                
                // Feature engineering
                new FeatureEngineeringStep(new FeatureEngineeringConfig
                {
                    GeneratePolynomialFeatures = true,
                    PolynomialDegree = 2,
                    GenerateInteractionFeatures = true,
                    MaxInteractionFeatures = 10
                }),
                
                // Normalization
                new NormalizationStep(NormalizationMethod.ZScore),
                
                // Custom transformation
                new CustomTransformationStep(
                    transform: data => ApplyLogTransform(data),
                    name: "LogTransform"),
                
                // Data augmentation
                new DataAugmentationStep(new DataAugmentationConfig
                {
                    AugmentationFactor = 2,
                    AddNoise = true,
                    NoiseLevel = 0.01
                })
            };
            
            // Execute pipeline
            var transformedData = features;
            var transformedTargets = targets;
            
            foreach (var step in pipeline)
            {
                Console.WriteLine($"Executing: {step.Name}");
                await step.FitAsync(transformedData, transformedTargets);
                transformedData = await step.TransformAsync(transformedData);
            }
            
            Console.WriteLine($"\nPipeline Result:");
            Console.WriteLine($"  Original shape: [{features.Length}, {features[0].Length}]");
            Console.WriteLine($"  Transformed shape: [{transformedData.Length}, {transformedData[0].Length}]");
        }
        
        // Helper methods
        
        private static (double[][] features, double[] targets) GenerateRegressionData(
            int samples, int features, bool drift = false)
        {
            var random = new Random(42);
            var data = new double[samples][];
            var targets = new double[samples];
            
            for (int i = 0; i < samples; i++)
            {
                data[i] = new double[features];
                double target = 0;
                
                for (int j = 0; j < features; j++)
                {
                    // Add drift if requested
                    double driftFactor = drift ? (1.0 + i * 0.01) : 1.0;
                    data[i][j] = (random.NextDouble() * 2 - 1) * driftFactor;
                    target += data[i][j] * (j + 1); // Linear combination
                }
                
                targets[i] = target + random.NextDouble() * 0.1; // Add noise
            }
            
            return (data, targets);
        }
        
        private static double GetModelSize(object model)
        {
            // Estimate model size in MB
            if (model is INeuralNetworkModel<double> nn)
            {
                var paramCount = nn.GetParameterCount();
                return paramCount * 8.0 / (1024 * 1024); // 8 bytes per double
            }
            return 0.1; // Default small size
        }
        
        private static double[][] ApplyLogTransform(double[][] data)
        {
            var transformed = new double[data.Length][];
            
            for (int i = 0; i < data.Length; i++)
            {
                transformed[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    // Apply log transform with offset to handle negative values
                    transformed[i][j] = Math.Log(Math.Abs(data[i][j]) + 1) * Math.Sign(data[i][j]);
                }
            }
            
            return transformed;
        }
    }
    
    /// <summary>
    /// Custom transformation pipeline step
    /// </summary>
    public class CustomTransformationStep : PipelineStepBase
    {
        private readonly Func<double[][], double[][]> transform;
        
        public CustomTransformationStep(Func<double[][], double[][]> transform, string name) 
            : base(name)
        {
            this.transform = transform ?? throw new ArgumentNullException(nameof(transform));
            IsCacheable = true;
        }
        
        protected override bool RequiresFitting() => false;
        
        protected override void FitCore(double[][] inputs, double[]? targets)
        {
            // No fitting required for custom transformation
            UpdateMetadata("InputShape", $"[{inputs.Length}, {inputs[0].Length}]");
        }
        
        protected override double[][] TransformCore(double[][] inputs)
        {
            var result = transform(inputs);
            UpdateMetadata("OutputShape", $"[{result.Length}, {result[0].Length}]");
            return result;
        }
    }
}