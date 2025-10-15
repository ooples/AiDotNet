using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Represents the final result of a federated learning training process.
    /// </summary>
    public class FederatedTrainingResult
    {
        /// <summary>
        /// Gets or sets the total number of rounds completed.
        /// </summary>
        public int TotalRounds { get; set; }

        /// <summary>
        /// Gets or sets the total time taken for the training process.
        /// </summary>
        public TimeSpan TrainingTime { get; set; }

        /// <summary>
        /// Gets or sets the final global model parameters.
        /// </summary>
        public Dictionary<string, Vector<double>> FinalGlobalParameters { get; set; }

        /// <summary>
        /// Gets or sets the convergence history across all rounds.
        /// </summary>
        public List<double> ConvergenceHistory { get; set; }

        /// <summary>
        /// Gets or sets the detailed training history for each round.
        /// </summary>
        public List<ServerRound> TrainingHistory { get; set; }

        /// <summary>
        /// Gets or sets whether the training converged successfully.
        /// </summary>
        public bool Converged { get; set; }

        /// <summary>
        /// Gets or sets the final convergence value.
        /// </summary>
        public double FinalConvergenceValue { get; set; }

        /// <summary>
        /// Gets or sets the reason for training termination.
        /// </summary>
        public string TerminationReason { get; set; }

        /// <summary>
        /// Gets or sets the total number of clients that participated.
        /// </summary>
        public int TotalParticipatingClients { get; set; }

        /// <summary>
        /// Gets or sets the average number of clients per round.
        /// </summary>
        public double AverageClientsPerRound { get; set; }

        /// <summary>
        /// Gets or sets additional metrics collected during training.
        /// </summary>
        public Dictionary<string, object> AdditionalMetrics { get; set; }

        /// <summary>
        /// Initializes a new instance of the FederatedTrainingResult class.
        /// </summary>
        public FederatedTrainingResult()
        {
            FinalGlobalParameters = new Dictionary<string, Vector<double>>();
            ConvergenceHistory = new List<double>();
            TrainingHistory = new List<ServerRound>();
            AdditionalMetrics = new Dictionary<string, object>();
            TerminationReason = "Unknown";
        }

        /// <summary>
        /// Calculates and updates derived statistics from the training history.
        /// </summary>
        public void CalculateStatistics()
        {
            if (TrainingHistory != null && TrainingHistory.Count > 0)
            {
                TotalRounds = TrainingHistory.Count;
                TotalParticipatingClients = TrainingHistory.Sum(r => r.ParticipatingClients);
                AverageClientsPerRound = (double)TotalParticipatingClients / TotalRounds;
                
                if (ConvergenceHistory != null && ConvergenceHistory.Count > 0)
                {
                    FinalConvergenceValue = ConvergenceHistory.Last();
                }
            }
        }

        /// <summary>
        /// Gets the average round time.
        /// </summary>
        /// <returns>The average time per round.</returns>
        public TimeSpan GetAverageRoundTime()
        {
            if (TrainingHistory == null || TrainingHistory.Count == 0)
                return TimeSpan.Zero;

            var totalTicks = TrainingHistory.Sum(r => r.RoundTime.Ticks);
            return TimeSpan.FromTicks(totalTicks / TrainingHistory.Count);
        }

        /// <summary>
        /// Gets the total communication overhead in bytes.
        /// </summary>
        /// <returns>The total communication bytes across all rounds.</returns>
        public long GetTotalCommunicationBytes()
        {
            if (TrainingHistory == null)
                return 0;

            return TrainingHistory.Sum(r => r.CommunicationBytes);
        }

        /// <summary>
        /// Checks if the training was successful.
        /// </summary>
        /// <returns>True if training completed successfully; otherwise, false.</returns>
        public bool IsSuccessful()
        {
            return Converged || 
                   (TerminationReason != null && 
                    (TerminationReason.IndexOf("completed", StringComparison.OrdinalIgnoreCase) >= 0 ||
                     TerminationReason.IndexOf("converged", StringComparison.OrdinalIgnoreCase) >= 0));
        }

        /// <summary>
        /// Generates a summary report of the training results.
        /// </summary>
        /// <returns>A formatted summary string.</returns>
        public string GenerateSummary()
        {
            CalculateStatistics();
            
            return $"Federated Training Summary:\n" +
                   $"- Total Rounds: {TotalRounds}\n" +
                   $"- Training Time: {TrainingTime.TotalMinutes:F1} minutes\n" +
                   $"- Converged: {(Converged ? "Yes" : "No")}\n" +
                   $"- Final Convergence: {FinalConvergenceValue:F6}\n" +
                   $"- Average Clients/Round: {AverageClientsPerRound:F1}\n" +
                   $"- Average Round Time: {GetAverageRoundTime().TotalSeconds:F1}s\n" +
                   $"- Termination Reason: {TerminationReason}";
        }

        /// <summary>
        /// Exports the convergence history as a CSV string.
        /// </summary>
        /// <returns>CSV formatted convergence history.</returns>
        public string ExportConvergenceHistoryAsCsv()
        {
            var csv = "Round,ConvergenceMetric\n";
            
            for (int i = 0; i < ConvergenceHistory.Count; i++)
            {
                csv += $"{i + 1},{ConvergenceHistory[i]}\n";
            }
            
            return csv;
        }
    }
}