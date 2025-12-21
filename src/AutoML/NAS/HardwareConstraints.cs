namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Hardware constraints for NAS.
    /// Defines maximum latency, energy, and memory limits for architecture search.
    /// </summary>
    /// <typeparam name="T">The numeric type for constraint values</typeparam>
    public class HardwareConstraints<T>
    {
        /// <summary>
        /// Gets or sets the maximum latency constraint in milliseconds. Null means no constraint.
        /// </summary>
        public double? MaxLatency { get; set; }

        /// <summary>
        /// Gets or sets the maximum energy constraint in millijoules. Null means no constraint.
        /// </summary>
        public double? MaxEnergy { get; set; }

        /// <summary>
        /// Gets or sets the maximum memory constraint in megabytes. Null means no constraint.
        /// </summary>
        public double? MaxMemory { get; set; }
    }
}

