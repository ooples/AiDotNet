using System.Collections.Generic;

namespace AiDotNet.Deployment.Techniques
{
    /// <summary>
    /// Options for post-training optimization.
    /// </summary>
    public class OptimizationOptions
    {
        /// <summary>
        /// Gets or sets the quantization strategy to use.
        /// </summary>
        public string Strategy { get; set; } = "int8";
        
        /// <summary>
        /// Gets or sets a value indicating whether to enable fine-tuning.
        /// </summary>
        public bool EnableFineTuning { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the target hardware for optimization.
        /// </summary>
        public string TargetHardware { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether to enable graph optimization.
        /// </summary>
        public bool EnableGraphOptimization { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the number of fine-tuning iterations.
        /// </summary>
        public int FineTuningIterations { get; set; } = 100;
        
        /// <summary>
        /// Gets or sets the fine-tuning learning rate.
        /// </summary>
        public float FineTuningLearningRate { get; set; } = 0.001f;
        
        /// <summary>
        /// Gets or sets custom optimization parameters.
        /// </summary>
        public Dictionary<string, object> CustomParameters { get; set; } = new Dictionary<string, object>();
    }
}