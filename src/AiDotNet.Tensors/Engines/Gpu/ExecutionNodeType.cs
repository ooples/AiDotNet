namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Defines the type of node in an execution graph.
/// </summary>
public enum ExecutionNodeType
{
    /// <summary>
    /// Buffer allocation node.
    /// Allocates GPU memory for a tensor.
    /// </summary>
    Allocate = 0,

    /// <summary>
    /// Host-to-device transfer node.
    /// Uploads data from CPU to GPU.
    /// </summary>
    TransferH2D = 1,

    /// <summary>
    /// Device-to-host transfer node.
    /// Downloads data from GPU to CPU.
    /// </summary>
    TransferD2H = 2,

    /// <summary>
    /// Single kernel execution node.
    /// Runs one GPU kernel.
    /// </summary>
    Kernel = 3,

    /// <summary>
    /// Fused kernel execution node.
    /// Runs a pre-fused sequence of kernels (e.g., GEMM+Bias+ReLU).
    /// </summary>
    FusedKernel = 4,

    /// <summary>
    /// Synchronization barrier node.
    /// Forces all prior operations to complete before continuing.
    /// </summary>
    Barrier = 5,

    /// <summary>
    /// Event record node.
    /// Records an event that can be waited on by other streams.
    /// </summary>
    RecordEvent = 6,

    /// <summary>
    /// Event wait node.
    /// Waits for an event recorded by another stream.
    /// </summary>
    WaitEvent = 7,

    /// <summary>
    /// Buffer deallocation node.
    /// Releases GPU memory.
    /// </summary>
    Deallocate = 8,

    /// <summary>
    /// Buffer copy node.
    /// Copies data between GPU buffers.
    /// </summary>
    Copy = 9,

    /// <summary>
    /// Reduction operation node.
    /// Performs reduction operations (sum, max, etc.).
    /// </summary>
    Reduction = 10,

    /// <summary>
    /// Normalization operation node.
    /// Batch norm, layer norm, etc.
    /// </summary>
    Normalization = 11,

    /// <summary>
    /// Attention operation node.
    /// Multi-head attention, flash attention, etc.
    /// </summary>
    Attention = 12,

    /// <summary>
    /// Convolution operation node.
    /// 2D/3D convolutions.
    /// </summary>
    Convolution = 13
}
