using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.MetaLearning.Models
{
    /// <summary>
    /// Represents a federated learning task for meta-learning
    /// </summary>
    public class FederatedTask
    {
        /// <summary>
        /// Unique identifier for the task
        /// </summary>
        public string TaskId { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Type or category of the task
        /// </summary>
        public string TaskType { get; set; } = string.Empty;

        /// <summary>
        /// Support set data for task adaptation
        /// </summary>
        public Matrix<double> SupportSet { get; set; } = new();

        /// <summary>
        /// Labels for the support set
        /// </summary>
        public Vector<double> SupportLabels { get; set; } = new();

        /// <summary>
        /// Query set data for task evaluation
        /// </summary>
        public Matrix<double> QuerySet { get; set; } = new();

        /// <summary>
        /// Labels for the query set
        /// </summary>
        public Vector<double> QueryLabels { get; set; } = new();

        /// <summary>
        /// Additional metadata about the task
        /// </summary>
        public Dictionary<string, object> TaskMetadata { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// Client ID that owns this task
        /// </summary>
        public string? ClientId { get; set; }

        /// <summary>
        /// Task creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Task difficulty or complexity score
        /// </summary>
        public double? DifficultyScore { get; set; }

        /// <summary>
        /// Validate the task data
        /// </summary>
        public void Validate()
        {
            if (SupportSet == null)
                throw new ArgumentNullException(nameof(SupportSet));
            
            if (SupportLabels == null)
                throw new ArgumentNullException(nameof(SupportLabels));
            
            if (QuerySet == null)
                throw new ArgumentNullException(nameof(QuerySet));
            
            if (QueryLabels == null)
                throw new ArgumentNullException(nameof(QueryLabels));
            
            if (SupportSet.Rows != SupportLabels.Length)
                throw new ArgumentException($"Support set rows ({SupportSet.Rows}) must match support labels length ({SupportLabels.Length})");
            
            if (QuerySet.Rows != QueryLabels.Length)
                throw new ArgumentException($"Query set rows ({QuerySet.Rows}) must match query labels length ({QueryLabels.Length})");
            
            if (SupportSet.Columns != QuerySet.Columns)
                throw new ArgumentException($"Support set columns ({SupportSet.Columns}) must match query set columns ({QuerySet.Columns})");
        }

        /// <summary>
        /// Get the total number of examples in the task
        /// </summary>
        public int TotalExamples => SupportSet.Rows + QuerySet.Rows;

        /// <summary>
        /// Get the feature dimension
        /// </summary>
        public int FeatureDimension => SupportSet.Columns;

        /// <summary>
        /// Create a copy of the task
        /// </summary>
        public FederatedTask Clone()
        {
            return new FederatedTask
            {
                TaskId = TaskId,
                TaskType = TaskType,
                SupportSet = new Matrix<double>(SupportSet),
                SupportLabels = new Vector<double>(SupportLabels),
                QuerySet = new Matrix<double>(QuerySet),
                QueryLabels = new Vector<double>(QueryLabels),
                TaskMetadata = new Dictionary<string, object>(TaskMetadata),
                ClientId = ClientId,
                CreatedAt = CreatedAt,
                DifficultyScore = DifficultyScore
            };
        }
    }
}