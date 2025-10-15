using AiDotNet.Enums;

namespace AiDotNet.Deployment
{
    /// <summary>
    /// Options for cloud deployment optimization
    /// </summary>
    public class CloudOptimizationOptions
    {
        /// <summary>
        /// Gets or sets the target cloud platform
        /// </summary>
        public CloudPlatform Platform { get; set; }
        
        /// <summary>
        /// Gets or sets whether to enable auto-scaling
        /// </summary>
        public bool EnableAutoScaling { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the minimum number of instances
        /// </summary>
        public int MinInstances { get; set; } = 1;
        
        /// <summary>
        /// Gets or sets the maximum number of instances
        /// </summary>
        public int MaxInstances { get; set; } = 10;
        
        /// <summary>
        /// Gets or sets whether to enable GPU acceleration
        /// </summary>
        public bool EnableGPU { get; set; } = false;
        
        /// <summary>
        /// Gets or sets whether to enable model caching
        /// </summary>
        public bool EnableCaching { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the target latency in milliseconds
        /// </summary>
        public int TargetLatencyMs { get; set; } = 100;
    }
}