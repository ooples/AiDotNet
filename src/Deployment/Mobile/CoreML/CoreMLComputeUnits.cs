namespace AiDotNet.Deployment.Mobile.CoreML;

/// <summary>
/// CoreML compute unit options.
/// </summary>
public enum CoreMLComputeUnits
{
    /// <summary>CPU only</summary>
    CPUOnly,

    /// <summary>CPU and GPU</summary>
    CPUAndGPU,

    /// <summary>All available units (CPU, GPU, Neural Engine)</summary>
    All,

    /// <summary>Neural Engine only (A11 and later)</summary>
    NeuralEngine
}
