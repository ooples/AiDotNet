namespace AiDotNet.FederatedLearning
{
    /// <summary>
    /// Communication settings for federated learning
    /// </summary>
    public class CommunicationSettings
    {
        public int TimeoutSeconds { get; set; } = 300;
        public int MaxRetries { get; set; } = 3;
        public bool UseCompression { get; set; } = true;
        public double CompressionRatio { get; set; } = 0.1;
        public bool UseBandwidthOptimization { get; set; } = true;
    }
}