using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents a fairness violation in model behavior
    /// </summary>
    public class FairnessViolation
    {
        /// <summary>
        /// Type of fairness metric violated
        /// </summary>
        public FairnessMetric MetricType { get; set; }
        
        /// <summary>
        /// Severity of the violation
        /// </summary>
        public string Severity { get; set; }
        
        /// <summary>
        /// Description of the violation
        /// </summary>
        public string Description { get; set; }
        
        /// <summary>
        /// Groups affected by the violation
        /// </summary>
        public List<string> AffectedGroups { get; set; }
        
        /// <summary>
        /// Initializes a new instance of FairnessViolation
        /// </summary>
        public FairnessViolation()
        {
            MetricType = FairnessMetric.DemographicParity;
            Severity = "Warning";
            Description = string.Empty;
            AffectedGroups = new List<string>();
        }
        
        /// <summary>
        /// Initializes a new instance with specified values
        /// </summary>
        public FairnessViolation(FairnessMetric metricType, string description)
        {
            MetricType = metricType;
            Severity = "Warning";
            Description = description;
            AffectedGroups = new List<string>();
        }
    }
}