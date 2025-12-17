namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Hardware constraints for NAS.
    /// </summary>
    public class HardwareConstraints<T>
    {
        public T? MaxLatency { get; set; }
        public T? MaxEnergy { get; set; }
        public T? MaxMemory { get; set; }
    }
}

