using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Aggregation
{
    /// <summary>
    /// Interface for parameter aggregation strategies in federated learning.
    /// </summary>
    public interface IParameterAggregator
    {
        /// <summary>
        /// Gets the aggregation strategy type.
        /// </summary>
        FederatedAggregationStrategy Strategy { get; }

        /// <summary>
        /// Aggregates parameters from multiple clients.
        /// </summary>
        /// <param name="clientUpdates">Dictionary of client updates, keyed by client ID.</param>
        /// <param name="clientWeights">Weights for each client in the aggregation.</param>
        /// <param name="strategy">The aggregation strategy to use.</param>
        /// <returns>Aggregated parameters as a dictionary.</returns>
        Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights,
            FederatedAggregationStrategy strategy);

        /// <summary>
        /// Validate client updates for consistency
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <returns>True if valid</returns>
        bool ValidateClientUpdates(Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates);

        /// <summary>
        /// Calculate aggregation quality metrics
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="aggregatedParameters">Aggregated parameters</param>
        /// <returns>Quality metrics</returns>
        AggregationMetrics CalculateAggregationMetrics(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, Vector<double>> aggregatedParameters);
    }
}