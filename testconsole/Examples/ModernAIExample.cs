namespace AiDotNet.Examples;

using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Optimizers;
using AiDotNet.Statistics;
using AiDotNet.Models.Inputs;
using AiDotNet.MockImplementations;
using AiDotNet.Pipeline.Steps;
using AiDotNet.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;

/// <summary>
/// Example demonstrating the modern AI features in PredictionModelBuilder.
/// </summary>
public static class ModernAiExample
{
    /// <summary>
    /// Demonstrates using AutoML to automatically find the best model.
    /// </summary>
    public static void AutoMLExample()
    {
        Console.WriteLine("=== AutoML Example ===");
        
        // Create sample data
        var features = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        
        var targets = new Vector<double>(new[] { 10.0, 25.0, 40.0, 55.0 });
        
        // Create builder with AutoML
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        
        // Enable AutoML with search constraints
        builder.EnableAutoML(new SimpleAutoMLModel<double>())
               .ConfigureAutoMLSearch(
                   new HyperparameterSearchSpace()
                       .AddContinuous("learning_rate", 0.001, 0.1)
                       .AddInteger("hidden_units", 10, 100)
                       .AddCategorical("activation", new[] { "relu", "tanh", "sigmoid" }),
                   timeLimit: TimeSpan.FromMinutes(5),
                   trialLimit: 50)
               .EnableNeuralArchitectureSearch(NeuralArchitectureSearchStrategy.Evolutionary);
        
        // Build model (AutoML will search for best architecture and hyperparameters)
        var model = builder.Build(features, targets);
        
        Console.WriteLine("AutoML search completed!");
    }
    
    /// <summary>
    /// Demonstrates using foundation models with few-shot learning.
    /// </summary>
    public static void FoundationModelExample()
    {
        Console.WriteLine("\n=== Foundation Model Example ===");
        
        // Create text classification data
        var texts = new Matrix<string>(new[,]
        {
            { "This movie is fantastic! Best I've seen all year." },
            { "Terrible film. Complete waste of time." },
            { "An okay movie, nothing special but watchable." }
        });
        
        var sentiments = new Vector<string>(new[] { "positive", "negative", "neutral" });
        
        // Create builder with foundation model
        var builder = new PredictionModelBuilder<float, Matrix<string>, Vector<string>>();
        
        // Use a pre-trained foundation model
        builder.UseFoundationModel(new BERTFoundationModel<float>())
               .ConfigureFineTuning(new FineTuningOptions<double>())
               .WithFewShotExamples(
                   (new Matrix<string>(new[,] { { "Amazing product!" } }), 
                    new Vector<string>(new[] { "positive" })),
                   (new Matrix<string>(new[,] { { "Disappointing quality." } }), 
                    new Vector<string>(new[] { "negative" }))
               );
        
        var model = builder.Build(texts, sentiments);
        
        Console.WriteLine("Foundation model fine-tuned successfully!");
    }
    
    /// <summary>
    /// Demonstrates multimodal AI combining text and images.
    /// </summary>
    public static void MultimodalExample()
    {
        Console.WriteLine("\n=== Multimodal AI Example ===");
        
        // Create multimodal data (simplified for example)
        var multimodalData = new MultimodalInput<double>()
            .AddTextData(new[] { "A red sports car", "A blue sedan", "A green SUV" })
            .AddImageData(new[] { "car1.jpg", "car2.jpg", "car3.jpg" });
        
        var prices = new Vector<double>(new[] { 50000.0, 25000.0, 35000.0 });
        
        // Create builder with multimodal model
        var builder = new PredictionModelBuilder<double, MultimodalInput<double>, Vector<double>>();
        
        builder.UseMultimodalModel(new CLIPMultimodalModel<double>())
               .AddModality(ModalityType.Text, new TextPreprocessor<double>())
               .AddModality(ModalityType.Image, new ImagePreprocessor<double>())
               .ConfigureModalityFusion(ModalityFusionStrategy.CrossAttention);
        
        var model = builder.Build(multimodalData, prices);
        
        Console.WriteLine("Multimodal model trained successfully!");
    }
    
    /// <summary>
    /// Demonstrates model interpretability features.
    /// </summary>
    public static void InterpretabilityExample()
    {
        Console.WriteLine("\n=== Interpretability Example ===");
        
        // Create sample data for credit scoring
        var features = new Matrix<double>(new double[,]
        {
            { 25.0, 50000.0, 700.0 }, // age, income, credit_score
            { 35.0, 75000.0, 650.0 },
            { 45.0, 100000.0, 800.0 },
            { 30.0, 40000.0, 600.0 }
        });
        
        var approved = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0 });
        
        // Create builder with interpretability
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.SetModel(new LogisticRegression<double>(new LogisticRegressionOptions<double>()));
        
        var interpretableWrapper = new InterpretableModelWrapper<double>();
        builder.WithInterpretability(interpretableWrapper)
               .EnableInterpretationMethods(
                   InterpretationMethod.SHAP,
                   InterpretationMethod.LIME,
                   InterpretationMethod.FeatureImportance)
               .ConfigureFairness(
                   sensitiveFeatures: new[] { 0 }, // age is sensitive
                   FairnessMetric.EqualOpportunity,
                   FairnessMetric.DemographicParity);
        
        var model = builder.Build(features, approved);
        
        // Get explanations
        // Note: In production, the builder would wrap the model with interpretability features
        // For this example, we'll demonstrate how to use the interpretable wrapper
        if (interpretableWrapper != null)
        {
            var importance = interpretableWrapper.GetGlobalFeatureImportanceAsync().Result;
            if (importance.Count >= 3)
            {
                Console.WriteLine("Feature importance: Age={0:F2}, Income={1:F2}, CreditScore={2:F2}",
                    importance[0], importance[1], importance[2]);
            }
        }
    }
    
    /// <summary>
    /// Demonstrates production monitoring and drift detection.
    /// </summary>
    public static void ProductionMonitoringExample()
    {
        Console.WriteLine("\n=== Production Monitoring Example ===");
        
        // Create initial training data
        var features = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 }, { 2.0, 3.0 }, { 3.0, 4.0 }, { 4.0, 5.0 }
        });
        var targets = new Vector<double>(new[] { 3.0, 5.0, 7.0, 9.0 });
        
        // Create builder with production monitoring
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.SetModel(new SimpleRegression<double>());
        
        var productionMonitor = new StandardProductionMonitor<double>();
        builder.WithProductionMonitoring(productionMonitor)
               .ConfigureDriftDetection(
                   dataDriftThreshold: 0.1,
                   conceptDriftThreshold: 0.15)
               .ConfigureAutoRetraining(
                   performanceDropThreshold: 0.2,
                   timeBasedRetraining: TimeSpan.FromDays(30));
        
        var model = builder.Build(features, targets);
        
        // Simulate production usage
        // Note: In production, the builder would integrate monitoring with the model
        // For this example, we'll demonstrate monitoring separately
        if (productionMonitor != null)
        {
            // New production data (potentially drifted)
            var newData = new Matrix<double>(new double[,]
            {
                { 5.0, 6.0 }, { 6.0, 7.0 } // Different distribution
            });
            
            var driftResult = productionMonitor.DetectDataDriftAsync(newData).Result;
            Console.WriteLine($"Data drift score: {driftResult.DriftScore:F3}");
            
            var recommendation = productionMonitor.GetRetrainingRecommendationAsync().Result;
            if (recommendation.ShouldRetrain)
            {
                Console.WriteLine("Model retraining recommended!");
            }
        }
    }
    
    /// <summary>
    /// Demonstrates cloud and edge optimization.
    /// </summary>
    public static void DeploymentOptimizationExample()
    {
        Console.WriteLine("\n=== Deployment Optimization Example ===");
        
        var features = new Matrix<double>(100, 10); // 100 samples, 10 features
        var targets = new Vector<double>(100);
        
        // Initialize with random data
        var random = new Random(42);
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                features[i, j] = random.NextDouble();
            }
            targets[i] = random.NextDouble();
        }
        
        // For cloud deployment
        var cloudBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        var cloudArchitecture = new NeuralNetworkArchitecture<double>(NetworkComplexity.Simple);
        cloudBuilder.SetModel(new NeuralNetworkModel<double>(cloudArchitecture));
        cloudBuilder.OptimizeForCloud(CloudPlatform.AWS, AiDotNet.Enums.OptimizationLevel.Aggressive);
        
        var cloudOptimizedModel = cloudBuilder.Build(features, targets);
        Console.WriteLine("Model optimized for AWS cloud deployment");
        
        // For edge deployment
        var edgeBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        var edgeArchitecture = new NeuralNetworkArchitecture<double>(NetworkComplexity.Simple);
        edgeBuilder.SetModel(new NeuralNetworkModel<double>(edgeArchitecture));
        edgeBuilder.OptimizeForEdge(
                       EdgeDevice.Smartphone,
                       memoryLimit: 50, // 50MB
                       latencyTarget: 10); // 10ms
        
        var edgeOptimizedModel = edgeBuilder.Build(features, targets);
        Console.WriteLine("Model optimized for mobile edge deployment");
    }
    
    /// <summary>
    /// Demonstrates federated learning setup.
    /// </summary>
    public static void FederatedLearningExample()
    {
        Console.WriteLine("\n=== Federated Learning Example ===");
        
        // Each client has their own local data
        var client1Data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 }, { 2.0, 3.0 }
        });
        var client1Targets = new Vector<double>(new[] { 3.0, 5.0 });
        
        // Create builder with federated learning
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.SetModel(new SimpleRegression<double>());
        
        builder.EnableFederatedLearning(
                   FederatedAggregationStrategy.SecureAggregation,
                   privacyBudget: 1.0) // Differential privacy budget
               .ConfigureMetaLearning(
                   Enums.MetaLearningAlgorithm.MAML,
                   innerLoopSteps: 5);
        
        // In real federated learning, this would be distributed
        var model = builder.Build(client1Data, client1Targets);
        
        Console.WriteLine("Federated learning model initialized");
    }
    
    /// <summary>
    /// Demonstrates advanced pipeline with branches.
    /// </summary>
    public static void AdvancedPipelineExample()
    {
        Console.WriteLine("\n=== Advanced Pipeline Example ===");
        
        var features = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targets = new Vector<double>(new[] { 10.0, 25.0, 40.0 });
        
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        
        // Add custom pipeline steps
        builder.AddPipelineStep(new LogTransformStep<double>(), PipelinePosition.PreProcessing)
               .AddPipelineStep(new PolynomialFeaturesStep<double>(degree: 2), PipelinePosition.FeatureEngineering)
               
               // Create branches for A/B testing
               .CreateBranch("modelA", b => {
                   b.SetModel(new SimpleRegression<double>());
                   b.ConfigureOptimizer(new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>()));
               })
               
               .CreateBranch("modelB", b => {
                   b.SetModel(new PolynomialRegression<double>(new PolynomialRegressionOptions<double>()));
                   b.ConfigureOptimizer(new AdamOptimizer<double, Matrix<double>, Vector<double>>(new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>()));
               })
               
               // Merge branches with ensemble
               .MergeBranches(BranchMergeStrategy.WeightedAverage, "modelA", "modelB");
        
        var model = builder.Build(features, targets);
        
        Console.WriteLine("Advanced pipeline with A/B testing completed");
    }
}