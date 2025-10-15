using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.MetaLearning.Models
{
    /// <summary>
    /// Result of client-side meta-learning adaptation
    /// </summary>
    public class ClientMetaResult
    {
        /// <summary>
        /// Client identifier
        /// </summary>
        public string ClientId { get; set; } = string.Empty;

        /// <summary>
        /// Number of adaptation steps performed
        /// </summary>
        public int AdaptationSteps { get; set; }

        /// <summary>
        /// Final loss on the support set after adaptation
        /// </summary>
        public double SupportLoss { get; set; }

        /// <summary>
        /// Loss on the query set after adaptation
        /// </summary>
        public double QueryLoss { get; set; }

        /// <summary>
        /// Accuracy on the task after adaptation
        /// </summary>
        public double TaskAccuracy { get; set; }

        /// <summary>
        /// Meta-gradients computed from the task
        /// </summary>
        public Dictionary<string, Vector<double>> MetaGradients { get; set; } = new Dictionary<string, Vector<double>>();

        /// <summary>
        /// Model parameters after adaptation
        /// </summary>
        public Dictionary<string, Vector<double>> AdaptedParameters { get; set; } = new Dictionary<string, Vector<double>>();

        /// <summary>
        /// Time taken for adaptation
        /// </summary>
        public TimeSpan AdaptationTime { get; set; }

        /// <summary>
        /// Initial loss before adaptation
        /// </summary>
        public double InitialLoss { get; set; }

        /// <summary>
        /// Improvement in loss from adaptation
        /// </summary>
        public double LossImprovement => InitialLoss - QueryLoss;

        /// <summary>
        /// Learning rate used for this task
        /// </summary>
        public double EffectiveLearningRate { get; set; }

        /// <summary>
        /// Whether adaptation converged
        /// </summary>
        public bool Converged { get; set; }

        /// <summary>
        /// Additional metrics collected during adaptation
        /// </summary>
        public Dictionary<string, double> AdditionalMetrics { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Create a summary string of the result
        /// </summary>
        public override string ToString()
        {
            return $"Client {ClientId}: Steps={AdaptationSteps}, " +
                   $"SupportLoss={SupportLoss:F4}, QueryLoss={QueryLoss:F4}, " +
                   $"Accuracy={TaskAccuracy:P2}, Converged={Converged}";
        }
    }
}