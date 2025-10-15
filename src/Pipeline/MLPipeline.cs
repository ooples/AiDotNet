using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Logging;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic machine learning pipeline that orchestrates multiple pipeline steps with production-ready features
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class MLPipeline<T> : IDisposable
    {
        private readonly List<IPipelineStep<T>> steps;
        private readonly Dictionary<string, object> pipelineMetadata;
        private readonly INumericOperations<T> numOps;
        private readonly ILogging? logger;
        private readonly SemaphoreSlim semaphore;
        private bool isFitted;
        private bool isDisposed;
        
        /// <summary>
        /// Gets the pipeline name
        /// </summary>
        public string Name { get; }
        
        /// <summary>
        /// Gets the unique pipeline ID
        /// </summary>
        public string PipelineId { get; }
        
        /// <summary>
        /// Gets whether the pipeline has been fitted
        /// </summary>
        public bool IsFitted => isFitted;
        
        /// <summary>
        /// Gets the pipeline steps
        /// </summary>
        public IReadOnlyList<IPipelineStep<T>> Steps => steps.AsReadOnly();
        
        /// <summary>
        /// Gets or sets the pipeline configuration
        /// </summary>
        public PipelineConfiguration Configuration { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the MLPipeline class
        /// </summary>
        /// <param name="name">Pipeline name</param>
        /// <param name="logger">Optional logger for pipeline operations</param>
        /// <param name="configuration">Optional pipeline configuration</param>
        public MLPipeline(string name = "MLPipeline", ILogging? logger = null, PipelineConfiguration? configuration = null)
        {
            Name = name;
            PipelineId = Guid.NewGuid().ToString();
            this.logger = logger;
            Configuration = configuration ?? new PipelineConfiguration();
            
            steps = new List<IPipelineStep<T>>();
            pipelineMetadata = new Dictionary<string, object>();
            numOps = MathHelper.GetNumericOperations<T>();
            semaphore = new SemaphoreSlim(1, 1);
            isFitted = false;
            isDisposed = false;
            
            InitializeMetadata();
        }
        
        /// <summary>
        /// Adds a step to the pipeline
        /// </summary>
        /// <param name="step">The pipeline step to add</param>
        /// <returns>The pipeline for method chaining</returns>
        public MLPipeline<T> AddStep(IPipelineStep<T> step)
        {
            if (step == null)
            {
                throw new ArgumentNullException(nameof(step));
            }
            
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(MLPipeline<T>));
            }
            
            steps.Add(step);
            logger?.Information($"Added step '{step.Name}' to pipeline '{Name}'");
            
            return this;
        }
        
        /// <summary>
        /// Removes a step from the pipeline
        /// </summary>
        /// <param name="stepName">Name of the step to remove</param>
        /// <returns>True if removed, false otherwise</returns>
        public bool RemoveStep(string stepName)
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(MLPipeline<T>));
            }
            
            var step = steps.FirstOrDefault(s => s.Name == stepName);
            if (step != null)
            {
                steps.Remove(step);
                logger?.Information($"Removed step '{stepName}' from pipeline '{Name}'");
                return true;
            }
            
            logger?.Warning($"Step '{stepName}' not found in pipeline '{Name}'");
            return false;
        }
        
        /// <summary>
        /// Fits the pipeline on the provided data
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target values (optional)</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task FitAsync(Matrix<T> inputs, Vector<T>? targets = null, CancellationToken cancellationToken = default)
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(MLPipeline<T>));
            }
            
            await semaphore.WaitAsync(cancellationToken);
            try
            {
                logger?.Information($"Starting pipeline '{Name}' fitting process");
                
                ValidatePipeline();
                ValidateInputData(inputs, targets);
                
                var stopwatch = Stopwatch.StartNew();
                var currentData = inputs;
                var currentTargets = targets;
                
                pipelineMetadata["FitStartTime"] = DateTime.UtcNow.ToString("O");
                
                // Fit each step in sequence
                for (int i = 0; i < steps.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    var step = steps[i];
                    var stepStopwatch = Stopwatch.StartNew();
                    
                    logger?.Information($"Fitting step {i + 1}/{steps.Count}: {step.Name}");
                    
                    try
                    {
                        await step.FitAsync(currentData, currentTargets);
                        
                        // Transform data for next step (except for the last step)
                        if (i < steps.Count - 1 && Configuration.TransformBetweenSteps)
                        {
                            currentData = await step.TransformAsync(currentData);
                            
                            // Handle steps that might modify targets
                            if (step is DataSplittingStep<T> splitter)
                            {
                                var (trainData, trainTargets) = splitter.GetTrainData(inputs, targets);
                                currentData = trainData;
                                currentTargets = trainTargets;
                            }
                        }
                        
                        stepStopwatch.Stop();
                        pipelineMetadata[$"Step_{step.Name}_FitTime"] = stepStopwatch.ElapsedMilliseconds;
                        
                        logger?.Information($"Step '{step.Name}' fitted in {stepStopwatch.ElapsedMilliseconds}ms");
                    }
                    catch (Exception ex)
                    {
                        logger?.Error($"Error fitting step '{step.Name}': {ex.Message}");
                        throw new InvalidOperationException($"Pipeline fitting failed at step '{step.Name}': {ex.Message}", ex);
                    }
                }
                
                isFitted = true;
                stopwatch.Stop();
                
                pipelineMetadata["FitEndTime"] = DateTime.UtcNow.ToString("O");
                pipelineMetadata["TotalFitTimeMs"] = stopwatch.ElapsedMilliseconds;
                pipelineMetadata["StepCount"] = steps.Count;
                
                logger?.Information($"Pipeline '{Name}' fitted successfully in {stopwatch.ElapsedMilliseconds}ms");
            }
            finally
            {
                semaphore.Release();
            }
        }
        
        /// <summary>
        /// Transforms data through the fitted pipeline
        /// </summary>
        /// <param name="inputs">Input data to transform</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Transformed data</returns>
        public async Task<Matrix<T>> TransformAsync(Matrix<T> inputs, CancellationToken cancellationToken = default)
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(MLPipeline<T>));
            }
            
            if (!isFitted)
            {
                throw new InvalidOperationException("Pipeline must be fitted before transformation");
            }
            
            await semaphore.WaitAsync(cancellationToken);
            try
            {
                var currentData = inputs;
                var stopwatch = Stopwatch.StartNew();
                
                logger?.Debug($"Starting pipeline '{Name}' transformation");
                
                // Transform through each step
                foreach (var step in steps)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    try
                    {
                        currentData = await step.TransformAsync(currentData);
                    }
                    catch (Exception ex)
                    {
                        logger?.Error($"Error transforming with step '{step.Name}': {ex.Message}");
                        throw new InvalidOperationException($"Pipeline transformation failed at step '{step.Name}': {ex.Message}", ex);
                    }
                }
                
                stopwatch.Stop();
                pipelineMetadata["LastTransformTimeMs"] = stopwatch.ElapsedMilliseconds;
                
                logger?.Debug($"Pipeline transformation completed in {stopwatch.ElapsedMilliseconds}ms");
                
                return currentData;
            }
            finally
            {
                semaphore.Release();
            }
        }
        
        /// <summary>
        /// Fits and transforms in a single operation
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target values (optional)</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Transformed data</returns>
        public async Task<Matrix<T>> FitTransformAsync(Matrix<T> inputs, Vector<T>? targets = null, CancellationToken cancellationToken = default)
        {
            await FitAsync(inputs, targets, cancellationToken);
            return await TransformAsync(inputs, cancellationToken);
        }
        
        /// <summary>
        /// Gets metadata about the pipeline
        /// </summary>
        /// <returns>Pipeline metadata</returns>
        public Dictionary<string, object> GetMetadata()
        {
            var metadata = new Dictionary<string, object>(pipelineMetadata)
            {
                ["IsFitted"] = isFitted,
                ["StepNames"] = string.Join(", ", steps.Select(s => s.Name)),
                ["IsDisposed"] = isDisposed,
                ["Configuration"] = Configuration
            };
            
            // Add individual step metadata
            foreach (var step in steps)
            {
                var stepMetadata = step.GetMetadata();
                foreach (var kvp in stepMetadata)
                {
                    metadata[$"{step.Name}_{kvp.Key}"] = kvp.Value;
                }
            }
            
            return metadata;
        }
        
        /// <summary>
        /// Gets a specific step by name
        /// </summary>
        /// <typeparam name="TStep">The type of step to get</typeparam>
        /// <param name="name">Step name</param>
        /// <returns>The step if found, null otherwise</returns>
        public TStep? GetStep<TStep>(string name) where TStep : IPipelineStep<T>
        {
            return steps.FirstOrDefault(s => s.Name == name && s is TStep) as TStep;
        }
        
        /// <summary>
        /// Validates the pipeline configuration
        /// </summary>
        private void ValidatePipeline()
        {
            if (steps.Count == 0)
            {
                throw new InvalidOperationException("Pipeline must contain at least one step");
            }
            
            // Validate step positions if they have position requirements
            var beginningSteps = steps.Where(s => s is PipelineStepBase<T> ps && ps.Position == PipelinePosition.Beginning).ToList();
            var endSteps = steps.Where(s => s is PipelineStepBase<T> ps && ps.Position == PipelinePosition.End).ToList();
            
            if (beginningSteps.Any() && steps.IndexOf(beginningSteps.First()) != 0)
            {
                throw new InvalidOperationException($"Step '{beginningSteps.First().Name}' must be at the beginning of the pipeline");
            }
            
            if (endSteps.Any() && steps.IndexOf(endSteps.First()) != steps.Count - 1)
            {
                throw new InvalidOperationException($"Step '{endSteps.First().Name}' must be at the end of the pipeline");
            }
            
            // Validate step compatibility
            if (Configuration.ValidateStepCompatibility)
            {
                for (int i = 0; i < steps.Count - 1; i++)
                {
                    // Add compatibility checks here if needed
                }
            }
        }
        
        /// <summary>
        /// Validates input data
        /// </summary>
        private void ValidateInputData(Matrix<T> inputs, Vector<T>? targets)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }
            
            if (inputs.Rows == 0 || inputs.Columns == 0)
            {
                throw new ArgumentException("Input matrix cannot be empty", nameof(inputs));
            }
            
            if (targets != null && targets.Length != inputs.Rows)
            {
                throw new ArgumentException($"Number of targets ({targets.Length}) must match number of input rows ({inputs.Rows})", nameof(targets));
            }
            
            if (Configuration.MaxInputSize > 0 && inputs.Rows * inputs.Columns > Configuration.MaxInputSize)
            {
                throw new ArgumentException($"Input size exceeds maximum allowed size of {Configuration.MaxInputSize}", nameof(inputs));
            }
        }
        
        /// <summary>
        /// Initializes pipeline metadata
        /// </summary>
        private void InitializeMetadata()
        {
            pipelineMetadata["PipelineId"] = PipelineId;
            pipelineMetadata["Name"] = Name;
            pipelineMetadata["CreatedAt"] = DateTime.UtcNow.ToString("O");
            pipelineMetadata["Version"] = "1.0.0";
            pipelineMetadata["NumericType"] = typeof(T).Name;
        }
        
        /// <summary>
        /// Creates a copy of the pipeline
        /// </summary>
        /// <returns>A new pipeline with the same steps</returns>
        public MLPipeline<T> Clone()
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(MLPipeline<T>));
            }
            
            var newPipeline = new MLPipeline<T>($"{Name}_Clone", logger, Configuration.Clone());
            
            foreach (var step in steps)
            {
                // Note: This assumes steps are immutable or implement proper cloning
                // In a production system, steps should implement ICloneable<T>
                newPipeline.AddStep(step);
            }
            
            return newPipeline;
        }
        
        /// <summary>
        /// Clears all steps from the pipeline
        /// </summary>
        public void Clear()
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(MLPipeline<T>));
            }
            
            steps.Clear();
            isFitted = false;
            InitializeMetadata();
            
            logger?.Information($"Pipeline '{Name}' cleared");
        }
        
        /// <summary>
        /// Disposes of the pipeline resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        
        /// <summary>
        /// Disposes of the pipeline resources
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!isDisposed)
            {
                if (disposing)
                {
                    semaphore?.Dispose();
                    
                    // Dispose of any disposable steps
                    foreach (var step in steps)
                    {
                        if (step is IDisposable disposableStep)
                        {
                            disposableStep.Dispose();
                        }
                    }
                }
                
                isDisposed = true;
            }
        }
    }
}
