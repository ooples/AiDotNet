namespace AiDotNet.AutoML
{
    /// <summary>
    /// Types of constraints for AutoML search
    /// </summary>
    public enum ConstraintType
    {
        /// <summary>
        /// Maximum model size in memory
        /// </summary>
        MaxModelSize,
        
        /// <summary>
        /// Maximum inference time for predictions
        /// </summary>
        MaxInferenceTime,
        
        /// <summary>
        /// Minimum accuracy requirement
        /// </summary>
        MinAccuracy,
        
        /// <summary>
        /// Maximum memory usage during training
        /// </summary>
        MaxMemoryUsage,
        
        /// <summary>
        /// Require the model to be interpretable
        /// </summary>
        RequireInterpretability
    }
}