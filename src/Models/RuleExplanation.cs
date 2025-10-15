using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents a rule-based explanation for model predictions
    /// </summary>
    public class RuleExplanation
    {
        /// <summary>
        /// Gets or sets the rule description
        /// </summary>
        public string Rule { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the confidence of the rule
        /// </summary>
        public double Confidence { get; set; }
        
        /// <summary>
        /// Gets or sets the support (number of instances covered)
        /// </summary>
        public int Support { get; set; }
        
        /// <summary>
        /// Gets or sets the conditions that make up the rule
        /// </summary>
        public List<string> Conditions { get; set; } = new List<string>();
    }
}