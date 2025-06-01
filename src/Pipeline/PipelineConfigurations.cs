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
        public Dictionary<string, object> ColumnConfigs { get; set; } = new Dictionary<string, object>();
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
        public List<string> DropColumns { get; set; } = new List<string>();
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
        public Dictionary<string, object> AugmentationParams { get; set; } = new Dictionary<string, object>();
    }
    
    /// <summary>
    /// AutoML configuration
    /// </summary>
    public class AutoMLConfig
    {
        public int TimeLimit { get; set; } = 3600; // seconds
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
    }
    
    /// <summary>
    /// Deployment configuration
    /// </summary>
    public class DeploymentConfig
    {
        public CloudPlatform CloudPlatform { get; set; } = CloudPlatform.AWS;
        public bool EnableAutoScaling { get; set; } = true;
        public int MinInstances { get; set; } = 1;
        public int MaxInstances { get; set; } = 10;
        public string Endpoint { get; set; }
        public Dictionary<string, string> SecurityConfig { get; set; } = new Dictionary<string, string>();
    }
    
    /// <summary>
    /// Monitoring configuration
    /// </summary>
    public class MonitoringConfig
    {
        public bool EnableDriftDetection { get; set; } = true;
        public bool EnablePerformanceMonitoring { get; set; } = true;
        public double AlertThreshold { get; set; } = 0.05;
        public int MonitoringInterval { get; set; } = 3600; // seconds
        public List<MetricType> MetricsToTrack { get; set; } = new List<MetricType>();
    }
}