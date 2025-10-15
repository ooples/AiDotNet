using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents a counterfactual explanation
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class CounterfactualExplanation<T>
    {
        private readonly INumericOperations<T> _ops;
        
        /// <summary>
        /// Original input values
        /// </summary>
        public Vector<T> OriginalInput { get; set; }
        
        /// <summary>
        /// Counterfactual input values
        /// </summary>
        public Vector<T> CounterfactualInput { get; set; }
        
        /// <summary>
        /// Indices of features that were changed
        /// </summary>
        public List<int> ChangedFeatures { get; set; }
        
        /// <summary>
        /// Changes made to each feature
        /// </summary>
        public Dictionary<int, T> FeatureChanges { get; set; }
        
        /// <summary>
        /// Distance between original and counterfactual
        /// </summary>
        public T Distance { get; set; }
        
        /// <summary>
        /// Whether the counterfactual achieves the desired outcome
        /// </summary>
        public bool IsValid { get; set; }
        
        /// <summary>
        /// Confidence score
        /// </summary>
        public T Confidence { get; set; }
        
        /// <summary>
        /// Initializes a new instance of CounterfactualExplanation
        /// </summary>
        public CounterfactualExplanation()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            OriginalInput = new Vector<T>(0);
            CounterfactualInput = new Vector<T>(0);
            ChangedFeatures = new List<int>();
            FeatureChanges = new Dictionary<int, T>();
            Distance = _ops.Zero;
            IsValid = false;
            Confidence = _ops.Zero;
        }
    }
}