using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents an anchor explanation (sufficient conditions for a prediction)
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class AnchorExplanation<T>
    {
        private readonly INumericOperations<T> _ops;
        
        /// <summary>
        /// The anchor rules (conditions)
        /// </summary>
        public List<AnchorRule<T>> Rules { get; set; }
        
        /// <summary>
        /// Precision of the anchor (probability that anchor holds)
        /// </summary>
        public T Precision { get; set; }
        
        /// <summary>
        /// Coverage of the anchor (fraction of instances it applies to)
        /// </summary>
        public T Coverage { get; set; }
        
        /// <summary>
        /// Features involved in the anchor
        /// </summary>
        public List<int> FeatureIndices { get; set; }
        
        /// <summary>
        /// Human-readable explanation
        /// </summary>
        public string TextExplanation { get; set; }
        
        /// <summary>
        /// Confidence in the anchor
        /// </summary>
        public T Confidence { get; set; }
        
        /// <summary>
        /// Number of samples used to compute the anchor
        /// </summary>
        public int SampleCount { get; set; }
        
        /// <summary>
        /// Initializes a new instance of AnchorExplanation
        /// </summary>
        public AnchorExplanation()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            Rules = new List<AnchorRule<T>>();
            Precision = _ops.Zero;
            Coverage = _ops.Zero;
            FeatureIndices = new List<int>();
            TextExplanation = string.Empty;
            Confidence = _ops.Zero;
            SampleCount = 0;
        }
    }
}