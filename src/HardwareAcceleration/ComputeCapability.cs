namespace AiDotNet.HardwareAcceleration
{
    /// <summary>
    /// Represents the compute capability of a GPU device
    /// </summary>
    public class ComputeCapability
    {
        /// <summary>
        /// Gets or sets the major version of the compute capability
        /// </summary>
        public int Major { get; set; }

        /// <summary>
        /// Gets or sets the minor version of the compute capability
        /// </summary>
        public int Minor { get; set; }

        /// <summary>
        /// Initializes a new instance of the ComputeCapability class
        /// </summary>
        public ComputeCapability()
        {
        }

        /// <summary>
        /// Initializes a new instance of the ComputeCapability class with specified version
        /// </summary>
        /// <param name="major">The major version</param>
        /// <param name="minor">The minor version</param>
        public ComputeCapability(int major, int minor)
        {
            Major = major;
            Minor = minor;
        }

        /// <summary>
        /// Returns a string representation of the compute capability
        /// </summary>
        /// <returns>A string in the format "major.minor"</returns>
        public override string ToString()
        {
            return $"{Major}.{Minor}";
        }

        /// <summary>
        /// Compares this compute capability with another
        /// </summary>
        /// <param name="other">The other compute capability to compare</param>
        /// <returns>True if the compute capabilities are equal</returns>
        public bool Equals(ComputeCapability other)
        {
            if (other == null) return false;
            return Major == other.Major && Minor == other.Minor;
        }

        /// <summary>
        /// Checks if this compute capability is greater than or equal to the specified version
        /// </summary>
        /// <param name="major">The major version to compare</param>
        /// <param name="minor">The minor version to compare</param>
        /// <returns>True if this compute capability is greater than or equal to the specified version</returns>
        public bool IsAtLeast(int major, int minor)
        {
            if (Major > major) return true;
            if (Major < major) return false;
            return Minor >= minor;
        }
    }
}