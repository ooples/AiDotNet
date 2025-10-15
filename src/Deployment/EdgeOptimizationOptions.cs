using AiDotNet.Enums;

namespace AiDotNet.Deployment
{
    /// <summary>
    /// Options for edge deployment optimization
    /// </summary>
    public class EdgeOptimizationOptions
    {
        /// <summary>
        /// Gets or sets the target edge device
        /// </summary>
        public EdgeDevice Device { get; set; }
        
        /// <summary>
        /// Gets or sets the memory limit in MB
        /// </summary>
        public int MemoryLimitMB { get; set; } = 256;
        
        /// <summary>
        /// Gets or sets whether to enable quantization
        /// </summary>
        public bool EnableQuantization { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the quantization type
        /// </summary>
        public QuantizationType QuantizationType { get; set; } = QuantizationType.Int8;
        
        /// <summary>
        /// Gets or sets whether to enable model pruning
        /// </summary>
        public bool EnablePruning { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the pruning threshold
        /// </summary>
        public double PruningThreshold { get; set; } = 0.01;
        
        /// <summary>
        /// Gets or sets the power consumption limit in watts
        /// </summary>
        public double PowerLimitWatts { get; set; } = 5.0;
    }
}