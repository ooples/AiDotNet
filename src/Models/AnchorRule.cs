using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents a single rule in an anchor explanation
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class AnchorRule<T>
    {
        private readonly INumericOperations<T> _ops;
        
        /// <summary>
        /// Feature index
        /// </summary>
        public int FeatureIndex { get; set; }
        
        /// <summary>
        /// Feature name (if available)
        /// </summary>
        public string FeatureName { get; set; }
        
        /// <summary>
        /// Rule type (e.g., "equals", "greater_than", "in_range")
        /// </summary>
        public string RuleType { get; set; }
        
        /// <summary>
        /// Value or threshold for the rule
        /// </summary>
        public T Value { get; set; }
        
        /// <summary>
        /// Upper bound (for range rules)
        /// </summary>
        public T? UpperBound { get; set; }
        
        /// <summary>
        /// Human-readable description
        /// </summary>
        public string Description { get; set; }
        
        /// <summary>
        /// Initializes a new instance of AnchorRule
        /// </summary>
        public AnchorRule()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            FeatureIndex = 0;
            FeatureName = string.Empty;
            RuleType = string.Empty;
            Value = _ops.Zero;
            UpperBound = null;
            Description = string.Empty;
        }
        
        /// <summary>
        /// Initializes a new instance with specified values
        /// </summary>
        public AnchorRule(int featureIndex, string ruleType, T value)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            FeatureIndex = featureIndex;
            FeatureName = string.Empty;
            RuleType = ruleType;
            Value = value;
            UpperBound = null;
            Description = string.Empty;
        }
    }
}