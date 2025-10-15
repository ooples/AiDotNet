using System;

namespace AiDotNet.FederatedLearning.MetaLearning.Models
{
    /// <summary>
    /// Historical record of a meta-learning round
    /// </summary>
    public class MetaLearningRound
    {
        /// <summary>
        /// Round number
        /// </summary>
        public int Round { get; set; }

        /// <summary>
        /// Number of participating tasks
        /// </summary>
        public int ParticipatingTasks { get; set; }

        /// <summary>
        /// Average number of adaptation steps across tasks
        /// </summary>
        public double AverageAdaptationSteps { get; set; }

        /// <summary>
        /// Average task accuracy after adaptation
        /// </summary>
        public double AverageTaskAccuracy { get; set; }

        /// <summary>
        /// Meta-loss (average query loss across tasks)
        /// </summary>
        public double MetaLoss { get; set; }

        /// <summary>
        /// Time taken for the round
        /// </summary>
        public TimeSpan RoundTime { get; set; }

        /// <summary>
        /// Timestamp when the round started
        /// </summary>
        public DateTime StartTime { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Timestamp when the round completed
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Number of failed tasks in this round
        /// </summary>
        public int FailedTasks { get; set; }

        /// <summary>
        /// Average improvement in loss from adaptation
        /// </summary>
        public double AverageLossImprovement { get; set; }

        /// <summary>
        /// Learning rate used in this round
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Whether the round completed successfully
        /// </summary>
        public bool Success { get; set; } = true;

        /// <summary>
        /// Error message if the round failed
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Additional statistics for the round
        /// </summary>
        public Dictionary<string, double> Statistics { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Mark the round as completed
        /// </summary>
        public void Complete()
        {
            EndTime = DateTime.UtcNow;
            if (StartTime != default && EndTime.HasValue)
            {
                RoundTime = EndTime.Value - StartTime;
            }
        }

        /// <summary>
        /// Get a summary of the round
        /// </summary>
        public override string ToString()
        {
            return $"Round {Round}: {ParticipatingTasks} tasks, " +
                   $"Loss={MetaLoss:F4}, Accuracy={AverageTaskAccuracy:P2}, " +
                   $"Time={RoundTime.TotalSeconds:F1}s, " +
                   $"Success={Success}";
        }
    }
}