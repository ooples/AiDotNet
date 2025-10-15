using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.Models
{
    /// <summary>
    /// LIME (Local Interpretable Model-agnostic Explanations) explanation result
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class LimeExplanation<T>
    {
        private readonly INumericOperations<T> _ops;
        
        public LimeExplanation()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            Intercept = _ops.Zero;
            LocalScore = _ops.Zero;
            Coverage = _ops.Zero;
        }
        
        /// <summary>
        /// Gets or sets the feature weights showing contribution of each feature
        /// </summary>
        public Dictionary<int, T> FeatureWeights { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the intercept of the local linear model
        /// </summary>
        public T Intercept { get; set; }
        
        /// <summary>
        /// Gets or sets the local fidelity score (how well the explanation fits locally)
        /// </summary>
        public T LocalScore { get; set; }
        
        /// <summary>
        /// Gets or sets the coverage of the explanation (fraction of similar instances explained)
        /// </summary>
        public T Coverage { get; set; }
    }
}