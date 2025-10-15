using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Logging;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Builder for creating ML pipelines fluently
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class MLPipelineBuilder<T>
    {
        private readonly MLPipeline<T> pipeline;
        private readonly ILogging? logger;
        
        /// <summary>
        /// Initializes a new instance of the MLPipelineBuilder class
        /// </summary>
        /// <param name="name">Pipeline name</param>
        /// <param name="logger">Optional logger</param>
        /// <param name="configuration">Optional pipeline configuration</param>
        public MLPipelineBuilder(string name = "MLPipeline", ILogging? logger = null, PipelineConfiguration? configuration = null)
        {
            this.logger = logger;
            pipeline = new MLPipeline<T>(name, logger, configuration);
        }
        
        /// <summary>
        /// Adds a data loading step
        /// </summary>
        /// <param name="source">Data source path</param>
        /// <param name="sourceType">Type of data source</param>
        /// <param name="options">Loading options</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithDataLoading(string source, DataSourceType sourceType, DataLoadingOptions? options = null)
        {
            pipeline.AddStep(new DataLoadingStep<T>(source, sourceType, options));
            return this;
        }
        
        /// <summary>
        /// Adds a data loading step with custom loader
        /// </summary>
        /// <param name="customLoader">Custom data loader function</param>
        /// <param name="options">Loading options</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithDataLoading(Func<Task<(Matrix<T> data, Vector<T> labels)>> customLoader, DataLoadingOptions? options = null)
        {
            pipeline.AddStep(new DataLoadingStep<T>(customLoader, options));
            return this;
        }
        
        /// <summary>
        /// Adds a data cleaning step
        /// </summary>
        /// <param name="config">Cleaning configuration</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithDataCleaning(DataCleaningConfig? config = null)
        {
            pipeline.AddStep(new DataCleaningStep<T>(config ?? new DataCleaningConfig()));
            return this;
        }
        
        /// <summary>
        /// Adds a feature engineering step
        /// </summary>
        /// <param name="config">Feature engineering configuration</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithFeatureEngineering(FeatureEngineeringConfig<T>? config = null)
        {
            pipeline.AddStep(new FeatureEngineeringStep<T>(config ?? new FeatureEngineeringConfig<T>()));
            return this;
        }
        
        /// <summary>
        /// Adds a normalization step
        /// </summary>
        /// <param name="method">Normalization method</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithNormalization(NormalizationMethod method = NormalizationMethod.MinMax)
        {
            pipeline.AddStep(new NormalizationStep<T>(method));
            return this;
        }
        
        /// <summary>
        /// Adds a data splitting step
        /// </summary>
        /// <param name="config">Splitting configuration</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithDataSplitting(DataSplittingConfig? config = null)
        {
            pipeline.AddStep(new DataSplittingStep<T>(config ?? new DataSplittingConfig()));
            return this;
        }
        
        /// <summary>
        /// Adds a model training step
        /// </summary>
        /// <param name="config">Training configuration</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithModelTraining(ModelTrainingConfig<T> config)
        {
            if (config == null)
            {
                throw new ArgumentNullException(nameof(config));
            }
            
            pipeline.AddStep(new ModelTrainingStep<T>(config));
            return this;
        }
        
        /// <summary>
        /// Adds a custom pipeline step
        /// </summary>
        /// <param name="step">Custom pipeline step</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> WithCustomStep(IPipelineStep<T> step)
        {
            if (step == null)
            {
                throw new ArgumentNullException(nameof(step));
            }
            
            pipeline.AddStep(step);
            return this;
        }
        
        /// <summary>
        /// Configures the pipeline
        /// </summary>
        /// <param name="configAction">Configuration action</param>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> Configure(Action<PipelineConfiguration> configAction)
        {
            if (configAction == null)
            {
                throw new ArgumentNullException(nameof(configAction));
            }
            
            configAction(pipeline.Configuration);
            return this;
        }
        
        /// <summary>
        /// Validates the pipeline configuration
        /// </summary>
        /// <returns>The builder for method chaining</returns>
        public MLPipelineBuilder<T> Validate()
        {
            if (pipeline.Steps.Count == 0)
            {
                throw new InvalidOperationException("Pipeline must contain at least one step");
            }
            
            // Add more validation as needed
            logger?.Information($"Pipeline '{pipeline.Name}' validated successfully with {pipeline.Steps.Count} steps");
            
            return this;
        }
        
        /// <summary>
        /// Builds the pipeline
        /// </summary>
        /// <returns>The configured ML pipeline</returns>
        public MLPipeline<T> Build()
        {
            Validate();
            
            logger?.Information($"Built pipeline '{pipeline.Name}' with {pipeline.Steps.Count} steps");
            
            return pipeline;
        }
        
        /// <summary>
        /// Creates a new builder from an existing pipeline
        /// </summary>
        /// <param name="existingPipeline">Existing pipeline to copy</param>
        /// <param name="newName">Name for the new pipeline</param>
        /// <returns>A new builder with the same steps</returns>
        public static MLPipelineBuilder<T> CreateFrom(MLPipeline<T> existingPipeline, string newName)
        {
            if (existingPipeline == null)
            {
                throw new ArgumentNullException(nameof(existingPipeline));
            }
            
            var builder = new MLPipelineBuilder<T>(newName);
            
            foreach (var step in existingPipeline.Steps)
            {
                builder.WithCustomStep(step);
            }
            
            return builder;
        }
    }
}