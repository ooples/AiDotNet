using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.MetaLearning.Models
{
    /// <summary>
    /// Result of a complete meta-learning round
    /// </summary>
    public class MetaLearningResult
    {
        /// <summary>
        /// Round number
        /// </summary>
        public int Round { get; set; }

        /// <summary>
        /// List of participating client IDs
        /// </summary>
        public List<string> ParticipatingClients { get; set; } = new List<string>();

        /// <summary>
        /// Individual client results
        /// </summary>
        public Dictionary<string, ClientMetaResult> ClientResults { get; set; } = new Dictionary<string, ClientMetaResult>();

        /// <summary>
        /// Aggregated meta-gradients
        /// </summary>
        public Dictionary<string, Vector<double>> MetaGradients { get; set; } = new Dictionary<string, Vector<double>>();

        /// <summary>
        /// Total time for the round
        /// </summary>
        public TimeSpan RoundTime { get; set; }

        /// <summary>
        /// Average task loss across all clients
        /// </summary>
        public double AverageTaskLoss { get; set; }

        /// <summary>
        /// L2 norm of the meta-gradients
        /// </summary>
        public double MetaGradientNorm { get; set; }

        /// <summary>
        /// Average accuracy across all tasks
        /// </summary>
        public double AverageAccuracy => ClientResults.Values.Average(r => r.TaskAccuracy);

        /// <summary>
        /// Standard deviation of task losses
        /// </summary>
        public double TaskLossStdDev
        {
            get
            {
                var losses = ClientResults.Values.Select(r => r.QueryLoss).ToArray();
                var mean = losses.Average();
                var variance = losses.Select(x => Math.Pow(x - mean, 2)).Average();
                return Math.Sqrt(variance);
            }
        }

        /// <summary>
        /// Number of converged tasks
        /// </summary>
        public int ConvergedTasks => ClientResults.Values.Count(r => r.Converged);

        /// <summary>
        /// Average adaptation steps needed
        /// </summary>
        public double AverageAdaptationSteps => ClientResults.Values.Average(r => r.AdaptationSteps);

        /// <summary>
        /// Total number of gradient computations
        /// </summary>
        public int TotalGradientComputations => ClientResults.Values.Sum(r => r.AdaptationSteps);

        /// <summary>
        /// Create a summary of the round
        /// </summary>
        public string GetSummary()
        {
            return $"Round {Round}: {ParticipatingClients.Count} clients, " +
                   $"Avg Loss={AverageTaskLoss:F4}, Avg Accuracy={AverageAccuracy:P2}, " +
                   $"Converged={ConvergedTasks}/{ClientResults.Count}, " +
                   $"Time={RoundTime.TotalSeconds:F1}s";
        }

        /// <summary>
        /// Get performance metrics as a dictionary
        /// </summary>
        public Dictionary<string, double> GetMetrics()
        {
            return new Dictionary<string, double>
            {
                ["round"] = Round,
                ["num_clients"] = ParticipatingClients.Count,
                ["avg_task_loss"] = AverageTaskLoss,
                ["avg_accuracy"] = AverageAccuracy,
                ["task_loss_stddev"] = TaskLossStdDev,
                ["meta_gradient_norm"] = MetaGradientNorm,
                ["converged_tasks"] = ConvergedTasks,
                ["avg_adaptation_steps"] = AverageAdaptationSteps,
                ["total_gradient_computations"] = TotalGradientComputations,
                ["round_time_seconds"] = RoundTime.TotalSeconds
            };
        }
    }
}