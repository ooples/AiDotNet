using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents SHAP (SHapley Additive exPlanations) values for model interpretation
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class ShapValues<T>
    {
        private readonly INumericOperations<T> _ops;
        
        /// <summary>
        /// Base value (expected model output)
        /// </summary>
        public T BaseValue { get; set; }
        
        /// <summary>
        /// SHAP values for each feature
        /// </summary>
        public Vector<T> Values { get; set; }
        
        /// <summary>
        /// Feature names (optional)
        /// </summary>
        public List<string> FeatureNames { get; set; }
        
        /// <summary>
        /// Output value (sum of base value and all SHAP values)
        /// </summary>
        public T OutputValue { get; set; }
        
        /// <summary>
        /// Initializes a new instance of ShapValues
        /// </summary>
        public ShapValues()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            BaseValue = _ops.Zero;
            Values = new Vector<T>(0);
            FeatureNames = new List<string>();
            OutputValue = _ops.Zero;
        }
        
        /// <summary>
        /// Initializes a new instance with given values
        /// </summary>
        public ShapValues(T baseValue, Vector<T> values)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            BaseValue = baseValue;
            Values = values;
            FeatureNames = new List<string>();
            
            // Calculate output value
            OutputValue = BaseValue;
            for (int i = 0; i < values.Length; i++)
            {
                OutputValue = _ops.Add(OutputValue, values[i]);
            }
        }
        
        /// <summary>
        /// Gets the most important features by absolute SHAP value
        /// </summary>
        /// <param name="topK">Number of top features to return</param>
        /// <returns>Indices of top features</returns>
        public int[] GetTopFeatures(int topK)
        {
            var indices = new int[Values.Length];
            var absValues = new T[Values.Length];
            
            for (int i = 0; i < Values.Length; i++)
            {
                indices[i] = i;
                absValues[i] = _ops.Abs(Values[i]);
            }
            
            // Sort by absolute SHAP value (descending)
            Array.Sort(absValues, indices, (a, b) => {
                if (_ops.GreaterThan(a, b)) return -1;
                if (_ops.LessThan(a, b)) return 1;
                return 0;
            });
            
            var result = new int[Math.Min(topK, indices.Length)];
            Array.Copy(indices, result, result.Length);
            return result;
        }
    }
}