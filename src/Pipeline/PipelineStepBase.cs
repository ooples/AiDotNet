using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Base class for all pipeline steps providing common functionality
    /// </summary>
    public abstract class PipelineStepBase : IPipelineStep
    {
        private bool _isFitted;
        private readonly Dictionary<string, object> _parameters;
        private readonly Dictionary<string, string> _metadata;

        /// <summary>
        /// Gets the name of this pipeline step
        /// </summary>
        public virtual string Name { get; protected set; }

        /// <summary>
        /// Gets whether this step is fitted/trained
        /// </summary>
        public bool IsFitted => _isFitted;

        /// <summary>
        /// Gets or sets the position of this step in the pipeline
        /// </summary>
        public PipelinePosition Position { get; set; } = PipelinePosition.Any;

        /// <summary>
        /// Gets or sets whether this step can be cached
        /// </summary>
        public bool IsCacheable { get; set; } = true;

        /// <summary>
        /// Gets or sets whether this step can run in parallel
        /// </summary>
        public bool SupportsParallelExecution { get; set; } = false;

        /// <summary>
        /// Gets or sets the timeout for this step in milliseconds
        /// </summary>
        public int TimeoutMilliseconds { get; set; } = 300000; // 5 minutes default

        /// <summary>
        /// Initializes a new instance of the PipelineStepBase class
        /// </summary>
        /// <param name="name">The name of this pipeline step</param>
        protected PipelineStepBase(string name)
        {
            Name = name ?? GetType().Name;
            _parameters = new Dictionary<string, object>();
            _metadata = new Dictionary<string, string>
            {
                ["CreatedAt"] = DateTime.UtcNow.ToString("O"),
                ["Version"] = "1.0.0"
            };
        }

        /// <summary>
        /// Fits/trains this pipeline step on the provided data
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional for unsupervised steps)</param>
        /// <returns>Task representing the asynchronous operation</returns>
        public virtual async Task FitAsync(double[][] inputs, double[]? targets = null)
        {
            ValidateInputsForFit(inputs, targets);
            
            await Task.Run(() =>
            {
                FitCore(inputs, targets);
                _isFitted = true;
                _metadata["LastFittedAt"] = DateTime.UtcNow.ToString("O");
            }).ConfigureAwait(false);
        }

        /// <summary>
        /// Transforms the input data
        /// </summary>
        /// <param name="inputs">Input data to transform</param>
        /// <returns>Transformed data</returns>
        public virtual async Task<double[][]> TransformAsync(double[][] inputs)
        {
            if (!_isFitted && RequiresFitting())
            {
                throw new InvalidOperationException($"Pipeline step '{Name}' must be fitted before transformation.");
            }

            ValidateInputsForTransform(inputs);

            return await Task.Run(() => TransformCore(inputs)).ConfigureAwait(false);
        }

        /// <summary>
        /// Fits and transforms in a single operation
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional)</param>
        /// <returns>Transformed data</returns>
        public virtual async Task<double[][]> FitTransformAsync(double[][] inputs, double[]? targets = null)
        {
            await FitAsync(inputs, targets).ConfigureAwait(false);
            return await TransformAsync(inputs).ConfigureAwait(false);
        }

        /// <summary>
        /// Gets the parameters of this pipeline step
        /// </summary>
        /// <returns>Dictionary of parameter names and values</returns>
        public virtual Dictionary<string, object> GetParameters()
        {
            return new Dictionary<string, object>(_parameters);
        }

        /// <summary>
        /// Sets the parameters of this pipeline step
        /// </summary>
        /// <param name="parameters">Dictionary of parameter names and values</param>
        public virtual void SetParameters(Dictionary<string, object> parameters)
        {
            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }

            foreach (var kvp in parameters)
            {
                SetParameter(kvp.Key, kvp.Value);
            }
        }

        /// <summary>
        /// Validates that this step can process the given input
        /// </summary>
        /// <param name="inputs">Input data to validate</param>
        /// <returns>True if valid, false otherwise</returns>
        public virtual bool ValidateInput(double[][] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                return false;
            }

            if (inputs.Any(row => row == null))
            {
                return false;
            }

            // Check for consistent dimensions
            var firstRowLength = inputs[0].Length;
            if (inputs.Any(row => row.Length != firstRowLength))
            {
                return false;
            }

            return ValidateInputCore(inputs);
        }

        /// <summary>
        /// Gets metadata about this pipeline step
        /// </summary>
        /// <returns>Metadata dictionary</returns>
        public virtual Dictionary<string, string> GetMetadata()
        {
            var metadata = new Dictionary<string, string>(_metadata)
            {
                ["IsFitted"] = _isFitted.ToString(),
                ["Position"] = Position.ToString(),
                ["IsCacheable"] = IsCacheable.ToString(),
                ["SupportsParallelExecution"] = SupportsParallelExecution.ToString()
            };

            return metadata;
        }

        /// <summary>
        /// Core fitting logic to be implemented by derived classes
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional)</param>
        protected abstract void FitCore(double[][] inputs, double[]? targets);

        /// <summary>
        /// Core transformation logic to be implemented by derived classes
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>Transformed data</returns>
        protected abstract double[][] TransformCore(double[][] inputs);

        /// <summary>
        /// Additional input validation logic to be optionally overridden by derived classes
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>True if valid, false otherwise</returns>
        protected virtual bool ValidateInputCore(double[][] inputs)
        {
            return true;
        }

        /// <summary>
        /// Indicates whether this step requires fitting before transformation
        /// </summary>
        /// <returns>True if fitting is required, false otherwise</returns>
        protected virtual bool RequiresFitting()
        {
            return true;
        }

        /// <summary>
        /// Sets a single parameter value
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="value">Parameter value</param>
        protected virtual void SetParameter(string name, object value)
        {
            _parameters[name] = value;
        }

        /// <summary>
        /// Gets a parameter value
        /// </summary>
        /// <typeparam name="TParam">The type of the parameter</typeparam>
        /// <param name="name">Parameter name</param>
        /// <param name="defaultValue">Default value if parameter not found</param>
        /// <returns>The parameter value</returns>
        protected TParam GetParameter<TParam>(string name, TParam defaultValue = default!)
        {
            if (_parameters.TryGetValue(name, out var value) && value is TParam typedValue)
            {
                return typedValue;
            }
            return defaultValue;
        }

        /// <summary>
        /// Validates inputs for the fit operation
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data</param>
        private void ValidateInputsForFit(double[][] inputs, double[]? targets)
        {
            if (!ValidateInput(inputs))
            {
                throw new ArgumentException($"Invalid input data for pipeline step '{Name}'", nameof(inputs));
            }

            if (targets != null && targets.Length != inputs.Length)
            {
                throw new ArgumentException($"Number of targets ({targets.Length}) must match number of input samples ({inputs.Length})", nameof(targets));
            }
        }

        /// <summary>
        /// Validates inputs for the transform operation
        /// </summary>
        /// <param name="inputs">Input data</param>
        private void ValidateInputsForTransform(double[][] inputs)
        {
            if (!ValidateInput(inputs))
            {
                throw new ArgumentException($"Invalid input data for pipeline step '{Name}'", nameof(inputs));
            }
        }

        /// <summary>
        /// Updates metadata with a key-value pair
        /// </summary>
        /// <param name="key">Metadata key</param>
        /// <param name="value">Metadata value</param>
        protected void UpdateMetadata(string key, string value)
        {
            _metadata[key] = value;
        }

        /// <summary>
        /// Resets the fitted state of this step
        /// </summary>
        protected void ResetFittedState()
        {
            _isFitted = false;
            _metadata.Remove("LastFittedAt");
        }
    }
}