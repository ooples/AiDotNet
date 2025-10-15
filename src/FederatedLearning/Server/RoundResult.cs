using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Represents the result of a single federated learning round execution.
    /// </summary>
    public class RoundResult
    {
        /// <summary>
        /// Gets or sets the round number.
        /// </summary>
        public int Round { get; set; }

        /// <summary>
        /// Gets or sets the list of client IDs selected for this round.
        /// </summary>
        public List<string> SelectedClients { get; set; }

        /// <summary>
        /// Gets or sets the list of client IDs that actually participated in this round.
        /// </summary>
        public List<string> ParticipatingClients { get; set; }

        /// <summary>
        /// Gets or sets the aggregated model parameters after this round.
        /// </summary>
        public Dictionary<string, Vector<double>> AggregatedParameters { get; set; }

        /// <summary>
        /// Gets or sets the convergence metric for this round.
        /// </summary>
        public double ConvergenceMetric { get; set; }

        /// <summary>
        /// Gets or sets the time taken to complete this round.
        /// </summary>
        public TimeSpan RoundTime { get; set; }

        /// <summary>
        /// Gets or sets the aggregation strategy used in this round.
        /// </summary>
        public string AggregationStrategy { get; set; }

        /// <summary>
        /// Gets or sets whether differential privacy was applied in this round.
        /// </summary>
        public bool DifferentialPrivacyApplied { get; set; }

        /// <summary>
        /// Gets or sets additional metrics collected during this round.
        /// </summary>
        public Dictionary<string, double> AdditionalMetrics { get; set; }

        /// <summary>
        /// Initializes a new instance of the RoundResult class.
        /// </summary>
        public RoundResult()
        {
            SelectedClients = new List<string>();
            ParticipatingClients = new List<string>();
            AggregatedParameters = new Dictionary<string, Vector<double>>();
            AdditionalMetrics = new Dictionary<string, double>();
            AggregationStrategy = "FederatedAveraging";
        }

        /// <summary>
        /// Gets the participation rate for this round.
        /// </summary>
        /// <returns>The percentage of selected clients that participated.</returns>
        public double GetParticipationRate()
        {
            if (SelectedClients == null || SelectedClients.Count == 0)
                return 0.0;

            return (double)ParticipatingClients.Count / SelectedClients.Count;
        }

        /// <summary>
        /// Gets the number of clients that dropped out during this round.
        /// </summary>
        /// <returns>The number of dropped clients.</returns>
        public int GetDroppedClientsCount()
        {
            return SelectedClients.Count - ParticipatingClients.Count;
        }

        /// <summary>
        /// Checks if the round was successful based on minimum participation threshold.
        /// </summary>
        /// <param name="minimumParticipationRate">The minimum required participation rate.</param>
        /// <returns>True if the round meets the minimum participation requirement.</returns>
        public bool IsSuccessful(double minimumParticipationRate = 0.5)
        {
            return GetParticipationRate() >= minimumParticipationRate;
        }

        /// <summary>
        /// Creates a summary of the round result.
        /// </summary>
        /// <returns>A string summary of the round result.</returns>
        public string GetSummary()
        {
            return $"Round {Round}: {ParticipatingClients.Count}/{SelectedClients.Count} clients participated. " +
                   $"Convergence: {ConvergenceMetric:F6}, Time: {RoundTime.TotalSeconds:F1}s, " +
                   $"Strategy: {AggregationStrategy}";
        }
    }
}