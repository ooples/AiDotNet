using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// Contains risk analysis results for a particular action.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class RiskAnalysisResult<T>
    {
        /// <summary>
        /// Gets or sets the expected return of the action.
        /// </summary>
        public T ExpectedReturn { get; set; } = default!;
        
        /// <summary>
        /// Gets or sets the variance of the return distribution.
        /// </summary>
        public T ReturnVariance { get; set; } = default!;
        
        /// <summary>
        /// Gets or sets the 95% Value at Risk.
        /// </summary>
        public T ValueAtRisk95 { get; set; } = default!;
        
        /// <summary>
        /// Gets or sets the 95% Conditional Value at Risk.
        /// </summary>
        public T ConditionalValueAtRisk95 { get; set; } = default!;
        
        /// <summary>
        /// Gets or sets the risk-adjusted return (accounting for risk preference).
        /// </summary>
        public T RiskAdjustedReturn { get; set; } = default!;
        
        /// <summary>
        /// Gets or sets the probability of achieving a positive return.
        /// </summary>
        public T ProbabilityOfPositiveReturn { get; set; } = default!;
        
        /// <summary>
        /// Gets or sets the full distribution of possible returns.
        /// </summary>
        public Vector<T> ValueDistribution { get; set; } = default!;
    }
}