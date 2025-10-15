using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Pipeline configuration
    /// </summary>
    public class PipelineConfiguration
    {
        public string Name { get; set; } = "MLPipeline";
        public bool EnableLogging { get; set; } = true;
        public bool EnableCheckpointing { get; set; } = true;
        public int MaxRetries { get; set; } = 3;
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
        
        /// <summary>
        /// Whether to transform data between steps during fitting
        /// </summary>
        public bool TransformBetweenSteps { get; set; } = true;
        
        /// <summary>
        /// Whether to validate step compatibility
        /// </summary>
        public bool ValidateStepCompatibility { get; set; } = true;
        
        /// <summary>
        /// Maximum allowed input size (0 for no limit)
        /// </summary>
        public int MaxInputSize { get; set; } = 0;
        
        /// <summary>
        /// Whether to enable parallel execution where possible
        /// </summary>
        public bool EnableParallelExecution { get; set; } = false;
        
        /// <summary>
        /// Default timeout for operations in milliseconds
        /// </summary>
        public int DefaultTimeoutMs { get; set; } = 300000; // 5 minutes
        
        /// <summary>
        /// Whether to enable detailed logging
        /// </summary>
        public bool EnableDetailedLogging { get; set; } = false;
        
        /// <summary>
        /// Creates a copy of the configuration
        /// </summary>
        public PipelineConfiguration Clone()
        {
            return new PipelineConfiguration
            {
                Name = Name,
                EnableLogging = EnableLogging,
                EnableCheckpointing = EnableCheckpointing,
                MaxRetries = MaxRetries,
                Metadata = new Dictionary<string, object>(Metadata),
                TransformBetweenSteps = TransformBetweenSteps,
                ValidateStepCompatibility = ValidateStepCompatibility,
                MaxInputSize = MaxInputSize,
                EnableParallelExecution = EnableParallelExecution,
                DefaultTimeoutMs = DefaultTimeoutMs,
                EnableDetailedLogging = EnableDetailedLogging
            };
        }
    }
    
    /// <summary>
    /// Data cleaning configuration
    /// </summary>
    public class DataCleaningConfig
    {
        public bool RemoveNulls { get; set; } = true;
        public bool RemoveDuplicates { get; set; } = true;
        public bool HandleOutliers { get; set; } = false;
        public OutlierDetectionMethod OutlierMethod { get; set; } = OutlierDetectionMethod.IQR;
        public double OutlierThreshold { get; set; } = 3.0;
        public bool HandleMissingValues { get; set; } = true;
        public bool RemoveRowsWithMissing { get; set; } = false;
        public ImputationStrategy ImputationStrategy { get; set; } = ImputationStrategy.Mean;
        public Dictionary<string, object> ColumnConfigs { get; set; } = new Dictionary<string, object>();
        public Dictionary<string, object> ImputationStrategy { get; set; } = new Dictionary<string, object>();
    }
    
    /// <summary>
    /// Feature engineering configuration
    /// </summary>
    public class FeatureEngineeringConfig
    {
        public bool AutoGenerate { get; set; } = false;
        public bool GeneratePolynomialFeatures { get; set; } = false;
        public int PolynomialDegree { get; set; } = 2;
        public bool GenerateInteractionFeatures { get; set; } = false;
        public bool OneHotEncode { get; set; } = true;
        public bool ScaleFeatures { get; set; } = true;
        public int MaxInteractionFeatures { get; set; } = 10;
        public List<string> DropColumns { get; set; } = new List<string>();
        public List<string> PolynomialFeatures { get; set; } = new List<string>();
        public List<string> InteractionFeatures { get; set; } = new List<string>();
        public bool GenerateTimeFeatures { get; set; } = false;
        public bool GenerateTextFeatures { get; set; } = false;
        public bool GenerateCategoricalFeatures { get; set; } = false;
    }
    
    /// <summary>
    /// Data augmentation configuration
    /// </summary>
    public class DataAugmentationConfig
    {
        public bool EnableImageAugmentation { get; set; } = false;
        public bool EnableTextAugmentation { get; set; } = false;
        public bool EnableTabularAugmentation { get; set; } = false;
        public double AugmentationRatio { get; set; } = 1.0;
        public int AugmentationFactor { get; set; } = 2;
        public bool AddNoise { get; set; } = true;
        public double NoiseLevel { get; set; } = 0.01;
        public bool RandomScaling { get; set; } = true;
        public double ScalingRange { get; set; } = 0.1;
        public bool FeatureDropout { get; set; } = false;
        public double DropoutRate { get; set; } = 0.1;
        public Dictionary<string, object> AugmentationParams { get; set; } = new Dictionary<string, object>();
    }
    
    /// <summary>
    /// AutoML configuration
    /// </summary>
    public class AutoMLConfig
    {
        public int TimeLimit { get; set; } = 3600; // seconds
        public int TrialLimit { get; set; } = 100; // maximum trials
        public OptimizationMode OptimizationMode { get; set; } = OptimizationMode.Balanced;
        public bool EnableEnsemble { get; set; } = true;
        public int MaxModels { get; set; } = 20;
        public double ValidationSplit { get; set; } = 0.2;
        public List<ModelType> ModelsToTry { get; set; } = new List<ModelType>();
    }
    
    /// <summary>
    /// Neural architecture search configuration
    /// </summary>
    public class NASConfig
    {
        public NeuralArchitectureSearchStrategy Strategy { get; set; } = NeuralArchitectureSearchStrategy.EvolutionarySearch;
        public int MaxLayers { get; set; } = 10;
        public int PopulationSize { get; set; } = 50;
        public int Generations { get; set; } = 100;
        public double ResourceBudget { get; set; } = 100.0;
        public int MaxTrials { get; set; } = 100;
        public Dictionary<string, object> SearchSpace { get; set; } = new Dictionary<string, object>();
    }
    
    /// <summary>
    /// Hyperparameter tuning configuration
    /// </summary>
    public class HyperparameterTuningConfig
    {
        public int MaxTrials { get; set; } = 50;
        public MetricType OptimizationMetric { get; set; } = MetricType.Accuracy;
        public bool UseEarlyStopping { get; set; } = true;
        public int EarlyStoppingPatience { get; set; } = 5;
        public Dictionary<string, object> ParameterSpace { get; set; } = new Dictionary<string, object>();
        public Dictionary<string, object> SearchSpace { get; set; } = new Dictionary<string, object>();
        public HyperparameterTuningStrategy TuningStrategy { get; set; } = HyperparameterTuningStrategy.RandomSearch;
    }
    
    /// <summary>
    /// Deployment configuration
    /// </summary>
    public class DeploymentConfig
    {
        public DeploymentTarget Target { get; set; } = DeploymentTarget.CloudDeployment;
        public CloudPlatform CloudPlatform { get; set; } = CloudPlatform.AWS;
        public bool EnableAutoScaling { get; set; } = true;
        public int MinInstances { get; set; } = 1;
        public int MaxInstances { get; set; } = 10;
        public string? Endpoint { get; set; }
        public Dictionary<string, string> SecurityConfig { get; set; } = new Dictionary<string, string>();
    }
    
    /// <summary>
    /// Monitoring configuration
    /// </summary>
    public class MonitoringConfig
    {
        public bool EnableDriftDetection { get; set; } = true;
        public bool EnablePerformanceMonitoring { get; set; } = true;
        public bool EnableAnomalyDetection { get; set; } = false;
        public double AlertThreshold { get; set; } = 0.05;
        public int MonitoringInterval { get; set; } = 3600; // seconds
        public List<MetricType> MetricsToTrack { get; set; } = new List<MetricType>();
        public List<MetricType> MetricsToMonitor { get; set; } = new List<MetricType>();
    }

    /// <summary>
    /// Ensemble configuration for combining multiple models
    /// </summary>
    public class EnsembleConfig
    {
        public List<ModelType> Models { get; set; } = new List<ModelType>();
        public EnsembleStrategy Strategy { get; set; } = EnsembleStrategy.Voting;
        public double[] ModelWeights { get; set; } = null;
        public bool UseWeightedVoting { get; set; } = false;
        public int MinModels { get; set; } = 2;
        public int MaxModels { get; set; } = 10;
    }
}
