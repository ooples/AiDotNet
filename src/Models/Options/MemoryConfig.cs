using System;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Memory allocation configuration
    /// </summary>
    public class MemoryConfig
    {
        /// <summary>
        /// Whether to enable memory optimization
        /// </summary>
        public bool EnableMemoryOptimization { get; set; } = true;

        /// <summary>
        /// Maximum memory usage in MB
        /// </summary>
        public int? MaxMemoryMB { get; set; }

        /// <summary>
        /// Whether to offload to CPU when GPU memory is full
        /// </summary>
        public bool EnableCPUOffload { get; set; } = false;

        /// <summary>
        /// Memory growth strategy for GPU
        /// </summary>
        public bool AllowMemoryGrowth { get; set; } = true;

        /// <summary>
        /// Pre-allocate memory percentage (0-100)
        /// </summary>
        public int PreAllocatePercent { get; set; } = 80;

        /// <summary>
        /// Validates the memory configuration
        /// </summary>
        public void Validate()
        {
            if (MaxMemoryMB.HasValue && MaxMemoryMB.Value <= 0)
            {
                throw new InvalidOperationException("MaxMemoryMB must be greater than 0");
            }

            if (PreAllocatePercent < 0 || PreAllocatePercent > 100)
            {
                throw new InvalidOperationException("PreAllocatePercent must be between 0 and 100");
            }
        }
    }
}