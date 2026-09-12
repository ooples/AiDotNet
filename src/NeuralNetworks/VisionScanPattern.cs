namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Defines the scan pattern used by the Vision Mamba model to convert 2D patch grids into 1D sequences.
/// </summary>
public enum VisionScanPattern
{
    /// <summary>
    /// Bidirectional scan: forward + reverse, used by the original Vision Mamba (Vim) paper.
    /// </summary>
    Bidirectional,

    /// <summary>
    /// Cross-scan: four directional scans (L→R, R→L, T→B, B→T), used by VMamba.
    /// </summary>
    CrossScan,

    /// <summary>
    /// Continuous/zigzag scan preserving spatial locality, used by PlainMamba.
    /// </summary>
    Continuous
}
