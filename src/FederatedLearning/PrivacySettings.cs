namespace AiDotNet.FederatedLearning
{
    /// <summary>
    /// Privacy settings for differential privacy
    /// </summary>
    public class PrivacySettings
    {
        public bool UseDifferentialPrivacy { get; set; } = true;
        public double Epsilon { get; set; } = 1.0;
        public double Delta { get; set; } = 1e-5;
        public double ClippingThreshold { get; set; } = 1.0;
        public double NoiseMultiplier { get; set; } = 1.0;
    }
}