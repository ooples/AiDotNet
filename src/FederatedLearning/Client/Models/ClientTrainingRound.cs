using System;

namespace AiDotNet.FederatedLearning.Client.Models
{
    /// <summary>
    /// Represents a single training round for a federated learning client
    /// </summary>
    public class ClientTrainingRound
    {
        /// <summary>
        /// Round number
        /// </summary>
        public int Round { get; set; }
        
        /// <summary>
        /// Average training loss for this round
        /// </summary>
        public double TrainingLoss { get; set; }
        
        /// <summary>
        /// Time taken for training in this round
        /// </summary>
        public TimeSpan TrainingTime { get; set; }
        
        /// <summary>
        /// Size of the training data used
        /// </summary>
        public int DataSize { get; set; }
        
        /// <summary>
        /// L2 norm of parameter updates
        /// </summary>
        public double ParameterUpdateNorm { get; set; }

        /// <summary>
        /// Number of epochs completed
        /// </summary>
        public int EpochsCompleted { get; set; }

        /// <summary>
        /// Validation loss if validation was performed
        /// </summary>
        public double? ValidationLoss { get; set; }

        /// <summary>
        /// Validation accuracy if validation was performed
        /// </summary>
        public double? ValidationAccuracy { get; set; }

        /// <summary>
        /// Number of batches processed
        /// </summary>
        public int BatchesProcessed { get; set; }

        /// <summary>
        /// Average batch processing time
        /// </summary>
        public TimeSpan AverageBatchTime { get; set; }

        /// <summary>
        /// Whether differential privacy was applied
        /// </summary>
        public bool PrivacyApplied { get; set; }

        /// <summary>
        /// Privacy epsilon value if applied
        /// </summary>
        public double? PrivacyEpsilon { get; set; }

        /// <summary>
        /// Learning rate used in this round
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Memory usage during training (in MB)
        /// </summary>
        public double? MemoryUsageMB { get; set; }

        /// <summary>
        /// Timestamp when the round started
        /// </summary>
        public DateTime StartTime { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Timestamp when the round ended
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Additional metrics for this round
        /// </summary>
        public Dictionary<string, double> AdditionalMetrics { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Complete the training round
        /// </summary>
        public void Complete()
        {
            EndTime = DateTime.UtcNow;
            if (StartTime != default && EndTime.HasValue)
            {
                TrainingTime = EndTime.Value - StartTime;
            }
        }

        /// <summary>
        /// Get a summary of the training round
        /// </summary>
        public override string ToString()
        {
            var summary = $"Round {Round}: Loss={TrainingLoss:F4}, Time={TrainingTime.TotalSeconds:F1}s, DataSize={DataSize}";
            if (ValidationLoss.HasValue)
            {
                summary += $", ValLoss={ValidationLoss.Value:F4}";
            }
            if (ValidationAccuracy.HasValue)
            {
                summary += $", ValAcc={ValidationAccuracy.Value:P2}";
            }
            return summary;
        }
    }
}