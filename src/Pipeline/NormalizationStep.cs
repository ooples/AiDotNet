using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic production-ready normalization pipeline step
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class NormalizationStep<T> : PipelineStepBase<T>
    {
        private readonly NormalizationMethod method;
        private INormalizer<T, Matrix<T>, Matrix<T>>? normalizer;
        
        public NormalizationStep(NormalizationMethod method) 
            : base("Normalization", MathHelper.GetNumericOperations<T>())
        {
            this.method = method;
            IsCacheable = true;
            SupportsParallelExecution = true;
        }
        
        private List<NormalizationParameters<T>>? normalizationParameters;
        
        protected override void FitCore(Matrix<T> inputs, Vector<T>? targets)
        {
            normalizer = NormalizerFactory<T, Matrix<T>, Matrix<T>>.CreateNormalizer(method);
            
            if (normalizer == null)
            {
                throw new InvalidOperationException($"Failed to create normalizer for method {method} and type {typeof(T).Name}");
            }
            
            // Fit is done during transform, so we just initialize here
            var (_, parameters) = normalizer.NormalizeInput(inputs);
            normalizationParameters = parameters;
            
            UpdateMetadata("NormalizationMethod", method.ToString());
            UpdateMetadata("FeatureCount", inputs.Columns.ToString());
            UpdateMetadata("SampleCount", inputs.Rows.ToString());
        }
        
        protected override Matrix<T> TransformCore(Matrix<T> inputs)
        {
            if (normalizer == null)
            {
                throw new InvalidOperationException("Normalizer has not been fitted");
            }
            
            var (normalizedData, _) = normalizer.NormalizeInput(inputs);
            return normalizedData;
        }
        
        /// <summary>
        /// Inverse transform to get back original scale
        /// </summary>
        /// <param name="normalized">Normalized data</param>
        /// <returns>Data in original scale</returns>
        public Matrix<T> InverseTransform(Matrix<T> normalized)
        {
            if (normalizer == null || normalizationParameters == null)
            {
                throw new InvalidOperationException("Normalizer has not been fitted");
            }
            
            // For input normalization, we need the first parameter set
            if (normalizationParameters.Count > 0)
            {
                return normalizer.Denormalize(normalized, normalizationParameters[0]);
            }
            
            throw new NotSupportedException($"Normalizer {method} does not support inverse transformation");
        }
        
        /// <summary>
        /// Gets the parameters of this pipeline step
        /// </summary>
        /// <returns>Dictionary of parameter names and values</returns>
        public override Dictionary<string, object> GetParameters()
        {
            var parameters = base.GetParameters();
            parameters["NormalizationMethod"] = method;
            
            if (normalizer != null && normalizer is IParameterizable<T, Matrix<T>, Matrix<T>> parameterizable)
            {
                var normParams = parameterizable.GetParameters();
                // Store the entire parameter vector as a single entry
                parameters["Normalizer_Parameters"] = normParams;
            }
            
            return parameters;
        }
    }
    
    /// <summary>
    /// Parameters used for normalization
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class NormalizationParameters<T>
    {
        /// <summary>
        /// The normalization method used
        /// </summary>
        public NormalizationMethod Method { get; set; }
        
        /// <summary>
        /// Number of features
        /// </summary>
        public int FeatureCount { get; set; }
        
        /// <summary>
        /// Mean values for each feature (for Z-score normalization)
        /// </summary>
        public Vector<T>? Means { get; set; }
        
        /// <summary>
        /// Standard deviation values for each feature (for Z-score normalization)
        /// </summary>
        public Vector<T>? StandardDeviations { get; set; }
        
        /// <summary>
        /// Min values for each feature (for Min-Max normalization)
        /// </summary>
        public Vector<T>? Mins { get; set; }
        
        /// <summary>
        /// Max values for each feature (for Min-Max normalization)
        /// </summary>
        public Vector<T>? Maxs { get; set; }
        
        /// <summary>
        /// Generic parameters dictionary for other normalization methods
        /// </summary>
        public Dictionary<string, object>? Parameters { get; set; }
    }
    
    /// <summary>
    /// Interface for normalizers that support inverse transformation
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">Input type</typeparam>
    /// <typeparam name="TOutput">Output type</typeparam>
    public interface IInverseTransformable<T, TInput, TOutput>
    {
        /// <summary>
        /// Inverse transform data back to original scale
        /// </summary>
        /// <param name="transformed">Transformed data</param>
        /// <returns>Original scale data</returns>
        TOutput InverseTransform(TInput transformed);
    }
}