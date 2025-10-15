using System;

namespace AiDotNet.Models
{
    /// <summary>
    /// Fine-tuning progress information
    /// </summary>
    public class FineTuningProgress
    {
        public int CurrentEpoch { get; set; }
        public int TotalEpochs { get; set; }
        public int CurrentStep { get; set; }
        public int TotalSteps { get; set; }
        public double TrainingLoss { get; set; }
        public double ValidationLoss { get; set; }
        public TimeSpan ElapsedTime { get; set; }
        public TimeSpan EstimatedTimeRemaining { get; set; }
        
        /// <summary>
        /// Gets the completion percentage (0-100)
        /// </summary>
        public double PercentComplete => TotalSteps > 0 ? (CurrentStep * 100.0 / TotalSteps) : 0;
    }
}