namespace AiDotNet.HardwareAcceleration
{
    /// <summary>
    /// Properties of a hardware accelerator device
    /// </summary>
    public class DeviceProperties
    {
        /// <summary>
        /// Gets or sets the device name
        /// </summary>
        public string DeviceName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the total memory size in bytes
        /// </summary>
        public long MemorySize { get; set; }

        /// <summary>
        /// Gets or sets the compute capability
        /// </summary>
        public ComputeCapability ComputeCapability { get; set; } = new ComputeCapability();

        /// <summary>
        /// Gets or sets the maximum number of threads per block
        /// </summary>
        public int MaxThreadsPerBlock { get; set; }

        /// <summary>
        /// Gets or sets the maximum block dimensions
        /// </summary>
        public int[] MaxBlockDimensions { get; set; } = new int[3];

        /// <summary>
        /// Gets or sets the maximum grid dimensions
        /// </summary>
        public int[] MaxGridDimensions { get; set; } = new int[3];

        /// <summary>
        /// Gets or sets the warp size
        /// </summary>
        public int WarpSize { get; set; }

        /// <summary>
        /// Gets or sets the shared memory per block in bytes
        /// </summary>
        public long SharedMemoryPerBlock { get; set; }

        /// <summary>
        /// Gets or sets the clock rate in KHz
        /// </summary>
        public int ClockRate { get; set; }

        /// <summary>
        /// Gets or sets the number of multiprocessors
        /// </summary>
        public int MultiprocessorCount { get; set; }

        /// <summary>
        /// Gets or sets the driver version
        /// </summary>
        public string DriverVersion { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets whether the device supports double precision
        /// </summary>
        public bool SupportsDoublePrecision { get; set; }

        /// <summary>
        /// Gets or sets whether the device supports tensor cores
        /// </summary>
        public bool SupportsTensorCores { get; set; }

        /// <summary>
        /// Gets or sets the maximum texture dimensions
        /// </summary>
        public int[] MaxTextureDimensions { get; set; } = new int[3];

        /// <summary>
        /// Gets or sets the total constant memory in bytes
        /// </summary>
        public long TotalConstantMemory { get; set; }

        /// <summary>
        /// Gets or sets whether the device can map host memory
        /// </summary>
        public bool CanMapHostMemory { get; set; }

        /// <summary>
        /// Gets or sets the memory bus width in bits
        /// </summary>
        public int MemoryBusWidth { get; set; }

        /// <summary>
        /// Gets or sets the L2 cache size in bytes
        /// </summary>
        public long L2CacheSize { get; set; }

        /// <summary>
        /// Gets or sets the maximum threads per multiprocessor
        /// </summary>
        public int MaxThreadsPerMultiprocessor { get; set; }
    }
}