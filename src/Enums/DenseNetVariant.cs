namespace AiDotNet.Enums;

/// <summary>
/// Specifies the DenseNet model variant.
/// </summary>
/// <remarks>
/// <para>
/// DenseNet (Densely Connected Convolutional Networks) variants differ in their depth and
/// computational requirements. Each variant has different numbers of layers per dense block.
/// </para>
/// <para>
/// <b>For Beginners:</b> The number in the variant name (e.g., DenseNet121) indicates the total
/// number of layers in the network. Higher numbers mean deeper networks with potentially better
/// accuracy but requiring more computation time and memory.
/// </para>
/// </remarks>
public enum DenseNetVariant
{
    /// <summary>
    /// DenseNet-121: [6, 12, 24, 16] layers per block (8M parameters).
    /// </summary>
    /// <remarks>
    /// The most commonly used variant, offering a good balance between accuracy and efficiency.
    /// </remarks>
    DenseNet121,

    /// <summary>
    /// DenseNet-169: [6, 12, 32, 32] layers per block (14M parameters).
    /// </summary>
    /// <remarks>
    /// A deeper variant with improved accuracy at the cost of more computation.
    /// </remarks>
    DenseNet169,

    /// <summary>
    /// DenseNet-201: [6, 12, 48, 32] layers per block (20M parameters).
    /// </summary>
    /// <remarks>
    /// A very deep variant for tasks requiring high accuracy.
    /// </remarks>
    DenseNet201,

    /// <summary>
    /// DenseNet-264: [6, 12, 64, 48] layers per block (33M parameters).
    /// </summary>
    /// <remarks>
    /// The deepest standard variant, offering maximum accuracy potential.
    /// </remarks>
    DenseNet264,

    /// <summary>
    /// Custom DenseNet variant for testing with minimal layers.
    /// </summary>
    /// <remarks>
    /// Use this variant for unit tests to minimize construction time.
    /// Typically uses [2, 2, 2, 2] block configuration with small growth rate.
    /// </remarks>
    Custom
}
