using System.Collections.Generic;

namespace AiDotNet.Deployment.Techniques
{
    /// <summary>
    /// Configuration for quantization.
    /// </summary>
    public class QuantizationConfig
    {
        /// <summary>
        /// Gets or sets the default quantization strategy.
        /// </summary>
        public string DefaultStrategy { get; set; } = "int8";
        
        /// <summary>
        /// Gets or sets a value indicating whether to validate accuracy after quantization.
        /// </summary>
        public bool ValidateAccuracy { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the maximum allowed accuracy drop (as a fraction).
        /// </summary>
        public float AccuracyThreshold { get; set; } = 0.01f; // 1% accuracy drop allowed
        
        /// <summary>
        /// Gets or sets a value indicating whether to use symmetric quantization.
        /// </summary>
        public bool SymmetricQuantization { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the number of calibration batches to use.
        /// </summary>
        public int CalibrationBatches { get; set; } = 100;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use per-channel quantization.
        /// </summary>
        public bool PerChannelQuantization { get; set; } = true;
        
        /// <summary>
        /// Gets or sets hardware-specific configuration options.
        /// </summary>
        public Dictionary<string, object> HardwareConfig { get; set; } = new Dictionary<string, object>();
    }
}