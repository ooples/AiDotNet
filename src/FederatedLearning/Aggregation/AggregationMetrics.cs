using System.Collections.Generic;

namespace AiDotNet.FederatedLearning.Aggregation
{
    /// <summary>
    /// Aggregation quality metrics for federated learning.
    /// </summary>
    /// <remarks>
    /// This class provides comprehensive metrics to evaluate the quality and performance
    /// of parameter aggregation in federated learning scenarios. It helps monitor
    /// the convergence behavior, client participation, and computational efficiency
    /// of the aggregation process.
    /// </remarks>
    public class AggregationMetrics
    {
        /// <summary>
        /// Gets or sets the average variance of parameters across all clients.
        /// </summary>
        /// <remarks>
        /// This metric indicates how much the parameter updates vary between different clients.
        /// High variance might suggest heterogeneous data distribution or divergent training patterns.
        /// </remarks>
        public double AverageParameterVariance { get; set; }

        /// <summary>
        /// Gets or sets the number of clients that participated in this aggregation round.
        /// </summary>
        /// <remarks>
        /// This helps track client availability and participation rates over time.
        /// </remarks>
        public int ParticipatingClients { get; set; }

        /// <summary>
        /// Gets or sets the total number of parameters aggregated.
        /// </summary>
        /// <remarks>
        /// This represents the size of the model being trained in federated manner.
        /// </remarks>
        public int ParameterCount { get; set; }

        /// <summary>
        /// Gets or sets the time taken to perform the aggregation in seconds.
        /// </summary>
        /// <remarks>
        /// This metric helps monitor the computational efficiency of the aggregation algorithm.
        /// </remarks>
        public double AggregationTime { get; set; }

        /// <summary>
        /// Gets or sets the variance for each parameter individually.
        /// </summary>
        /// <remarks>
        /// This detailed breakdown helps identify which specific parameters show high variance
        /// across clients, which can be useful for debugging or optimization.
        /// </remarks>
        public Dictionary<string, double> PerParameterVariance { get; set; } = new Dictionary<string, double>();
    }
}