namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Configuration for a BigNAS sub-network.
    /// </summary>
    public class BigNASConfig
    {
        public int Depth { get; set; }
        public double WidthMultiplier { get; set; }
        public int KernelSize { get; set; }
        public int ExpansionRatio { get; set; }
        public int Resolution { get; set; }
        public bool IsTeacher { get; set; }
    }
}

