namespace AiDotNet.Deployment
{
    /// <summary>
    /// Configuration for mobile platform deployment
    /// </summary>
    public class MobilePlatformConfig
    {
        public string PlatformName { get; set; }
        public double MaxModelSize { get; set; } // MB
        public string[] SupportedFormats { get; set; }
        public string MinOSVersion { get; set; }
        public string[] HardwareAccelerators { get; set; }
    }
}