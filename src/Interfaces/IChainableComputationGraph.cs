using AiDotNet.Autodiff;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for layers that can build computation graphs with a provided input node.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This interface extends the JIT compilation capability by allowing composite layers
/// (layers that contain other layers) to properly chain computation graphs together.
/// Instead of creating their own input nodes, implementing layers can accept an input
/// node from a parent layer and build their computation on top of it.
/// </para>
/// <para><b>For Beginners:</b> When you have layers inside layers (like a DenseBlock
/// containing multiple DenseBlockLayers), the parent layer needs to connect the child
/// layers' computation graphs together. This interface provides that capability.
///
/// Without this interface:
/// - Each layer creates its own input node
/// - Parent can't properly chain child layers together
/// - Computation graphs are disconnected
///
/// With this interface:
/// - Parent provides its output as input to child
/// - Child builds computation on top of that input
/// - Computation graphs are properly connected
///
/// Example usage in a parent layer:
/// <code>
/// // Parent builds graph through children
/// var currentOutput = inputNode;
/// foreach (var childLayer in _childLayers)
/// {
///     currentOutput = childLayer.BuildComputationGraph(currentOutput, $"child{i}_");
/// }
/// return currentOutput;
/// </code>
/// </para>
/// </remarks>
public interface IChainableComputationGraph<T>
{
    /// <summary>
    /// Builds a computation graph using the provided input node.
    /// </summary>
    /// <param name="inputNode">The input computation node to build upon.</param>
    /// <param name="namePrefix">Prefix for node names to ensure uniqueness when multiple
    /// instances of the same layer type exist in a parent layer.</param>
    /// <returns>The output computation node representing this layer's computation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds the layer's computation graph using an externally provided input node,
    /// rather than creating its own input node. This allows parent composite layers to chain
    /// multiple sub-layers together into a single connected computation graph.
    /// </para>
    /// <para><b>For Beginners:</b> This is like giving the layer its input and asking it to
    /// compute its output. The layer doesn't need to worry about where the input came from -
    /// it just processes it and returns the result.
    ///
    /// The namePrefix parameter is important for debugging and visualization. When the same
    /// layer type appears multiple times (e.g., multiple conv layers in a block), the prefix
    /// helps distinguish them (e.g., "layer0_conv1", "layer1_conv1").
    /// </para>
    /// </remarks>
    ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix);
}
