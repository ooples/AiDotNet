using System.Collections.Generic;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents a decision rule extracted from a model
    /// </summary>
    public class RuleExplanation
    {
        /// <summary>
        /// The rule in human-readable format
        /// </summary>
        public string Rule { get; set; } = string.Empty;

        /// <summary>
        /// Confidence level of the rule (0-1)
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Number of samples supporting this rule
        /// </summary>
        public int Support { get; set; }

        /// <summary>
        /// Individual conditions that make up the rule
        /// </summary>
        public List<string> Conditions { get; set; } = new List<string>();

        /// <summary>
        /// The predicted outcome when this rule applies
        /// </summary>
        public double Outcome { get; set; }

        /// <summary>
        /// Feature indices involved in this rule
        /// </summary>
        public List<int> InvolvedFeatures { get; set; } = new List<int>();
    }
}