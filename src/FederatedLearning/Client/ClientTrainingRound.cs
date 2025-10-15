using System;

namespace AiDotNet.FederatedLearning.Client
{
    /// <summary>
    /// Client training round information
    /// </summary>
    public class ClientTrainingRound
    {
        /// <summary>
        /// Gets or sets the round number
        /// </summary>
        public int Round { get; set; }
        
        /// <summary>
        /// Gets or sets the training loss for this round
        /// </summary>
        public double TrainingLoss { get; set; }
        
        /// <summary>
        /// Gets or sets the time taken for training
        /// </summary>
        public TimeSpan TrainingTime { get; set; }
        
        /// <summary>
        /// Gets or sets the size of data used in training
        /// </summary>
        public int DataSize { get; set; }
        
        /// <summary>
        /// Gets or sets the norm of parameter updates (optional)
        /// </summary>
        public double? ParameterUpdateNorm { get; set; }
    }
}