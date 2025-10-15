namespace AiDotNet.FederatedLearning
{
    /// <summary>
    /// Security settings for federated learning
    /// </summary>
    public class FederatedSecuritySettings
    {
        public bool UseSecureAggregation { get; set; } = true;
        public bool UseEncryption { get; set; } = true;
        public bool VerifyClientIdentity { get; set; } = true;
        public int KeySize { get; set; } = 2048;
        public string EncryptionAlgorithm { get; set; } = "RSA";
    }
}