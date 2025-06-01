using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.AutoML;
using AiDotNet.ProductionMonitoring;
using AiDotNet.Deployment;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Fluent API for building ML pipelines
    /// </summary>
    public class MLPipelineBuilder
    {
        private readonly PipelineConfiguration configuration;
        private readonly List<IPipelineStep> steps;
        private DataLoadingStep dataLoadingStep;
        private ModelTrainingStep modelTrainingStep;
        private DeploymentStep deploymentStep;
        
        private MLPipelineBuilder()
        {
            configuration = new PipelineConfiguration();
            steps = new List<IPipelineStep>();
        }
        
        /// <summary>
        /// Create a new ML pipeline builder
        /// </summary>
        public static MLPipelineBuilder Create(string name = "MLPipeline")
        {
            return new MLPipelineBuilder
            {
                configuration = { Name = name }
            };
        }
        
        /// <summary>
        /// Load data from various sources
        /// </summary>
        public MLPipelineBuilder LoadData(string source, DataSourceType sourceType = DataSourceType.CSV)
        {
            dataLoadingStep = new DataLoadingStep(source, sourceType);
            steps.Add(dataLoadingStep);
            return this;
        }
        
        /// <summary>
        /// Load data with custom loader
        /// </summary>
        public MLPipelineBuilder LoadData(Func<(Tensor<double> data, Tensor<double> labels)> dataLoader)
        {
            dataLoadingStep = new DataLoadingStep(dataLoader);
            steps.Add(dataLoadingStep);
            return this;
        }
        
        /// <summary>
        /// Add data cleaning step
        /// </summary>
        public MLPipelineBuilder CleanData(Action<DataCleaningConfig> configure = null)
        {
            var config = new DataCleaningConfig();
            configure?.Invoke(config);
            steps.Add(new DataCleaningStep(config));
            return this;
        }
        
        /// <summary>
        /// Add feature engineering step
        /// </summary>
        public MLPipelineBuilder FeatureEngineering(Action<FeatureEngineeringConfig> configure = null)
        {
            var config = new FeatureEngineeringConfig();
            configure?.Invoke(config);
            steps.Add(new FeatureEngineeringStep(config));
            return this;
        }
        
        /// <summary>
        /// Split data into train/validation/test sets
        /// </summary>
        public MLPipelineBuilder SplitData(double trainRatio = 0.7, double valRatio = 0.15, double testRatio = 0.15)
        {
            steps.Add(new DataSplittingStep(trainRatio, valRatio, testRatio));
            return this;
        }
        
        /// <summary>
        /// Add data augmentation
        /// </summary>
        public MLPipelineBuilder AugmentData(Action<DataAugmentationConfig> configure)
        {
            var config = new DataAugmentationConfig();
            configure(config);
            steps.Add(new DataAugmentationStep(config));
            return this;
        }
        
        /// <summary>
        /// Add normalization
        /// </summary>
        public MLPipelineBuilder Normalize(NormalizationMethod method = NormalizationMethod.ZScore)
        {
            steps.Add(new NormalizationStep(method));
            return this;
        }
        
        /// <summary>
        /// Train a model with specified type
        /// </summary>
        public MLPipelineBuilder TrainModel(ModelType modelType, Action<ModelTrainingConfig> configure = null)
        {
            var config = new ModelTrainingConfig { ModelType = modelType };
            configure?.Invoke(config);
            modelTrainingStep = new ModelTrainingStep(config);
            steps.Add(modelTrainingStep);
            return this;
        }
        
        /// <summary>
        /// Use AutoML to find best model
        /// </summary>
        public MLPipelineBuilder AutoML(Action<AutoMLConfig> configure = null)
        {
            var config = new AutoMLConfig();
            configure?.Invoke(config);
            modelTrainingStep = new AutoMLStep(config);
            steps.Add(modelTrainingStep);
            return this;
        }
        
        /// <summary>
        /// Use Neural Architecture Search
        /// </summary>
        public MLPipelineBuilder NeuralArchitectureSearch(Action<NASConfig> configure = null)
        {
            var config = new NASConfig();
            configure?.Invoke(config);
            modelTrainingStep = new NASStep(config);
            steps.Add(modelTrainingStep);
            return this;
        }
        
        /// <summary>
        /// Add model ensemble
        /// </summary>
        public MLPipelineBuilder Ensemble(params ModelType[] models)
        {
            var config = new EnsembleConfig { Models = models.ToList() };
            modelTrainingStep = new EnsembleStep(config);
            steps.Add(modelTrainingStep);
            return this;
        }
        
        /// <summary>
        /// Add cross-validation
        /// </summary>
        public MLPipelineBuilder CrossValidate(CrossValidationType type = CrossValidationType.KFold, int folds = 5)
        {
            steps.Add(new CrossValidationStep(type, folds));
            return this;
        }
        
        /// <summary>
        /// Evaluate model with specified metrics
        /// </summary>
        public MLPipelineBuilder Evaluate(params MetricType[] metrics)
        {
            steps.Add(new EvaluationStep(metrics));
            return this;
        }
        
        /// <summary>
        /// Add hyperparameter tuning
        /// </summary>
        public MLPipelineBuilder TuneHyperparameters(Action<HyperparameterTuningConfig> configure)
        {
            var config = new HyperparameterTuningConfig();
            configure(config);
            steps.Add(new HyperparameterTuningStep(config));
            return this;
        }
        
        /// <summary>
        /// Add model interpretability
        /// </summary>
        public MLPipelineBuilder AddInterpretability(params InterpretationMethod[] methods)
        {
            steps.Add(new InterpretabilityStep(methods));
            return this;
        }
        
        /// <summary>
        /// Compress model for deployment
        /// </summary>
        public MLPipelineBuilder CompressModel(CompressionTechnique technique = CompressionTechnique.Quantization)
        {
            steps.Add(new ModelCompressionStep(technique));
            return this;
        }
        
        /// <summary>
        /// Deploy model to target
        /// </summary>
        public MLPipelineBuilder Deploy(DeploymentTarget target, Action<DeploymentConfig> configure = null)
        {
            var config = new DeploymentConfig { Target = target };
            configure?.Invoke(config);
            deploymentStep = new DeploymentStep(config);
            steps.Add(deploymentStep);
            return this;
        }
        
        /// <summary>
        /// Add production monitoring
        /// </summary>
        public MLPipelineBuilder Monitor(Action<MonitoringConfig> configure = null)
        {
            var config = new MonitoringConfig();
            configure?.Invoke(config);
            steps.Add(new MonitoringStep(config));
            return this;
        }
        
        /// <summary>
        /// Add A/B testing
        /// </summary>
        public MLPipelineBuilder ABTest(string experimentName, double trafficSplit = 0.5)
        {
            steps.Add(new ABTestingStep(experimentName, trafficSplit));
            return this;
        }
        
        /// <summary>
        /// Add custom step
        /// </summary>
        public MLPipelineBuilder AddCustomStep(IPipelineStep step)
        {
            steps.Add(step);
            return this;
        }
        
        /// <summary>
        /// Configure pipeline options
        /// </summary>
        public MLPipelineBuilder WithOptions(Action<PipelineOptions> configure)
        {
            configure(configuration.Options);
            return this;
        }
        
        /// <summary>
        /// Build the pipeline
        /// </summary>
        public MLPipeline<double, Matrix<double>, Vector<double>> Build()
        {
            ValidatePipeline();
            return new MLPipeline<double, Matrix<double>, Vector<double>>(configuration, steps);
        }
        
        /// <summary>
        /// Build and run the pipeline
        /// </summary>
        public async Task<PipelineResult> BuildAndRunAsync()
        {
            var pipeline = Build();
            return await pipeline.RunAsync();
        }
        
        /// <summary>
        /// Build and run the pipeline synchronously
        /// </summary>
        public PipelineResult BuildAndRun()
        {
            var pipeline = Build();
            return pipeline.Run();
        }
        
        private void ValidatePipeline()
        {
            if (dataLoadingStep == null)
                throw new InvalidOperationException("Pipeline must include a data loading step");
            
            if (modelTrainingStep == null)
                throw new InvalidOperationException("Pipeline must include a model training step");
        }
    }
}