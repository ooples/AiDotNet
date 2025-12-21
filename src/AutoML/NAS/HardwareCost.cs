namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Represents the hardware cost of an operation or architecture.
    /// </summary>
    public class HardwareCost<T>
    {
        public T Latency { get; set; } = default!;
        public T Energy { get; set; } = default!;
        public T Memory { get; set; } = default!;
    }
}

