namespace AiDotNet.Enums;

/// <summary>
/// Describes the Forward method signature shape a layer uses.
/// The test scaffold generator uses this to select the correct test base class.
/// </summary>
public enum LayerApiShape
{
    /// <summary>
    /// Standard Forward(Tensor) → Tensor interface.
    /// Used by the vast majority of layers (Dense, Conv, BatchNorm, Dropout, Pooling, etc.).
    /// </summary>
    SingleTensor,

    /// <summary>
    /// Dual-input Forward(Tensor, Tensor) → Tensor interface.
    /// Used by layers requiring a secondary input (CrossAttention, TransformerDecoder,
    /// MemoryRead/Write, UNetDiscriminator with skip connections).
    /// </summary>
    DualTensor,

    /// <summary>
    /// Multi-input Forward(params Tensor[]) → Tensor interface.
    /// Used by merge layers that combine multiple inputs (AddLayer, ConcatenateLayer, MultiplyLayer).
    /// </summary>
    MultiInput,

    /// <summary>
    /// Standard Forward(Tensor) but requires graph setup before use (adjacency matrix,
    /// Laplacian, eigenbasis, etc.). The test base provides synthetic graph data.
    /// Used by DiffusionConvLayer, MeshEdgeConvLayer, MeshPoolLayer, SpiralConvLayer.
    /// </summary>
    GraphWithSetup
}
