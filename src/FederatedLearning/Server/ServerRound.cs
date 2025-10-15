using System;

namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Represents information about a single training round in federated learning.
    /// </summary>
    public class ServerRound
    {
        /// <summary>
        /// Gets or sets the round number.
        /// </summary>
        public int Round { get; set; }

        /// <summary>
        /// Gets or sets the number of clients that participated in this round.
        /// </summary>
        public int ParticipatingClients { get; set; }

        /// <summary>
        /// Gets or sets the convergence metric for this round.
        /// </summary>
        public double ConvergenceMetric { get; set; }

        /// <summary>
        /// Gets or sets the time taken to complete this round.
        /// </summary>
        public TimeSpan RoundTime { get; set; }

        /// <summary>
        /// Gets or sets the L2 norm of the global parameters after this round.
        /// </summary>
        public double GlobalParameterNorm { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when this round started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when this round completed.
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Gets or sets the average client training time for this round.
        /// </summary>
        public TimeSpan AverageClientTrainingTime { get; set; }

        /// <summary>
        /// Gets or sets the number of clients that failed in this round.
        /// </summary>
        public int FailedClients { get; set; }

        /// <summary>
        /// Gets or sets the communication overhead in bytes for this round.
        /// </summary>
        public long CommunicationBytes { get; set; }

        /// <summary>
        /// Initializes a new instance of the ServerRound class.
        /// </summary>
        public ServerRound()
        {
            StartTime = DateTime.UtcNow;
            RoundTime = TimeSpan.Zero;
            AverageClientTrainingTime = TimeSpan.Zero;
        }

        /// <summary>
        /// Marks the round as completed and calculates the round time.
        /// </summary>
        public void Complete()
        {
            EndTime = DateTime.UtcNow;
            RoundTime = EndTime - StartTime;
        }

        /// <summary>
        /// Calculates the success rate for this round.
        /// </summary>
        /// <returns>The percentage of clients that successfully participated.</returns>
        public double GetSuccessRate()
        {
            var totalClients = ParticipatingClients + FailedClients;
            return totalClients > 0 ? (double)ParticipatingClients / totalClients : 0.0;
        }

        /// <summary>
        /// Returns a string representation of the server round.
        /// </summary>
        /// <returns>A string containing round information.</returns>
        public override string ToString()
        {
            return $"Round {Round}: {ParticipatingClients} clients, " +
                   $"Convergence={ConvergenceMetric:F6}, Time={RoundTime.TotalSeconds:F1}s";
        }
    }
}