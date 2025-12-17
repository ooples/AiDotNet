using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Configuration for an AttentiveNAS sub-network.
    /// </summary>
    public class AttentiveNASConfig<T>
    {
        public int Depth { get; set; }
        public double WidthMultiplier { get; set; }
        public int KernelSize { get; set; }
        public Vector<T> Embedding { get; set; } = null!;
    }
}

