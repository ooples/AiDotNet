namespace AiDotNet.FederatedLearning.Communication.Models
{
    /// <summary>
    /// Settings for federated learning communication
    /// </summary>
    public class CommunicationSettings
    {
        /// <summary>
        /// Timeout for communication operations in seconds
        /// </summary>
        public int TimeoutSeconds { get; set; } = 300;

        /// <summary>
        /// Maximum number of retry attempts
        /// </summary>
        public int MaxRetries { get; set; } = 3;

        /// <summary>
        /// Whether to use message compression
        /// </summary>
        public bool UseCompression { get; set; } = true;

        /// <summary>
        /// Whether to use message encryption
        /// </summary>
        public bool UseEncryption { get; set; } = true;

        /// <summary>
        /// Target compression ratio
        /// </summary>
        public double CompressionRatio { get; set; } = 0.1;

        /// <summary>
        /// Whether to use bandwidth optimization
        /// </summary>
        public bool UseBandwidthOptimization { get; set; } = true;

        /// <summary>
        /// Maximum message size in bytes
        /// </summary>
        public int MaxMessageSize { get; set; } = 10000000; // 10MB

        /// <summary>
        /// Whether to simulate network failures for testing
        /// </summary>
        public bool SimulateNetworkFailures { get; set; } = false;

        /// <summary>
        /// Simulated network latency in milliseconds
        /// </summary>
        public int SimulatedLatencyMs { get; set; } = 100;

        /// <summary>
        /// Server endpoint URL
        /// </summary>
        public string ServerEndpoint { get; set; } = string.Empty;

        /// <summary>
        /// Client endpoint URL
        /// </summary>
        public string ClientEndpoint { get; set; } = string.Empty;

        /// <summary>
        /// Authentication token
        /// </summary>
        public string AuthToken { get; set; } = string.Empty;

        /// <summary>
        /// Whether to use SSL/TLS
        /// </summary>
        public bool UseSecureConnection { get; set; } = true;
    }
}