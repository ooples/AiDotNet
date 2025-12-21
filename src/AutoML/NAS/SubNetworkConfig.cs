namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Configuration for a sub-network sampled from OFA.
    /// </summary>
    public class SubNetworkConfig
    {
        public int Depth { get; set; }
        public int KernelSize { get; set; }
        public double WidthMultiplier { get; set; }
        public int ExpansionRatio { get; set; }
    }
}

