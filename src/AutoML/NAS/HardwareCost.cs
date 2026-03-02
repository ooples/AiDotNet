namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Represents the hardware cost of an operation or architecture.
    /// </summary>
    public class HardwareCost<T>
    {
        public required T Latency { get; set; }
        public required T Energy { get; set; }
        public required T Memory { get; set; }
    }
}

