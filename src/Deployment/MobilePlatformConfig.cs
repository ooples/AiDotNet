using System;

namespace AiDotNet.Deployment
{
    /// <summary>
    /// Configuration for mobile platform deployment
    /// </summary>
    public class MobilePlatformConfig
    {
        public string PlatformName { get; set; } = string.Empty;
        public double MaxModelSize { get; set; } // MB
        public string[] SupportedFormats { get; set; } = new string[0];
        public string MinOSVersion { get; set; } = string.Empty;
        public string[] HardwareAccelerators { get; set; } = new string[0];
    }
}
