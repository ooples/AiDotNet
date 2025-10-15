using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Enums;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents fairness metrics for model evaluation
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class FairnessMetrics<T>
    {
        private readonly INumericOperations<T> _ops;
        
        /// <summary>
        /// Demographic parity difference
        /// </summary>
        public T DemographicParityDifference { get; set; }
        
        /// <summary>
        /// Equal opportunity difference
        /// </summary>
        public T EqualOpportunityDifference { get; set; }
        
        /// <summary>
        /// Equalized odds difference
        /// </summary>
        public T EqualizedOddsDifference { get; set; }
        
        /// <summary>
        /// Disparate impact ratio
        /// </summary>
        public T DisparateImpactRatio { get; set; }
        
        /// <summary>
        /// Statistical parity difference
        /// </summary>
        public T StatisticalParityDifference { get; set; }
        
        /// <summary>
        /// Group-specific metrics
        /// </summary>
        public Dictionary<string, GroupMetrics<T>> GroupMetrics { get; set; }
        
        /// <summary>
        /// Overall fairness score (0-1, higher is fairer)
        /// </summary>
        public T OverallFairnessScore { get; set; }
        
        /// <summary>
        /// Fairness violations found
        /// </summary>
        public List<FairnessViolation> Violations { get; set; }
        
        /// <summary>
        /// Initializes a new instance of FairnessMetrics
        /// </summary>
        public FairnessMetrics()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            DemographicParityDifference = _ops.Zero;
            EqualOpportunityDifference = _ops.Zero;
            EqualizedOddsDifference = _ops.Zero;
            DisparateImpactRatio = _ops.One;
            StatisticalParityDifference = _ops.Zero;
            GroupMetrics = new Dictionary<string, GroupMetrics<T>>();
            OverallFairnessScore = _ops.One;
            Violations = new List<FairnessViolation>();
        }
    }
}