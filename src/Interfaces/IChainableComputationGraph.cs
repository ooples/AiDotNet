using AiDotNet.Autodiff;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for layers that can chain their computation graph with a provided input node.
/// </summary>
/// <remarks>
/// <para>
/// This interface is designed for composite layers (layers that contain other layers internally)
/// to support JIT compilation. Instead of creating their own input variable node, implementing
/// layers build their computation graph using a provided input node, allowing parent layers
/// to chain multiple sub-layer computation graphs together.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some neural network layers are "composite" - they internally use other
/// layers (like convolutions, batch normalization, etc.) to perform their function. When we want
/// to compile the entire network for faster execution (JIT compilation), we need a way to connect
/// these internal layers together. This interface provides that connection point.
/// </para>
/// <para>
/// Example usage in a composite layer like DenseBlock:
/// <code>
/// public override ComputationNode&lt;T&gt; ExportComputationGraph(List&lt;ComputationNode&lt;T&gt;&gt; inputNodes)
/// {
///     var inputNode = TensorOperations&lt;T&gt;.Variable(symbolicInput, "input");
///     inputNodes.Add(inputNode);
///
///     var currentFeatures = inputNode;
///     foreach (var layer in _subLayers)
///     {
///         var layerOutput = layer.BuildComputationGraph(currentFeatures, $"layer{i}_");
///         currentFeatures = TensorOperations&lt;T&gt;.Concat(new[] { currentFeatures, layerOutput }, axis: 1);
///     }
///     return currentFeatures;
/// }
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ChainableComputationGraph")]
public interface IChainableComputationGraph<T>
{
    /// <summary>
    /// Builds the computation graph for this layer using the provided input node.
    /// </summary>
    /// <param name="inputNode">The input computation node from the parent layer.</param>
    /// <param name="namePrefix">Prefix for naming internal nodes (for debugging/visualization).</param>
    /// <returns>The output computation node representing this layer's computation.</returns>
    /// <remarks>
    /// <para>
    /// Unlike <see cref="ILayer{T}.ExportComputationGraph"/>, this method does NOT create a new
    /// input variable. Instead, it uses the provided <paramref name="inputNode"/> as its input,
    /// allowing the parent layer to chain multiple sub-layers together in a single computation graph.
    /// </para>
    /// <para>
    /// The <paramref name="namePrefix"/> parameter should be used to prefix all internal node names
    /// to avoid naming conflicts when multiple instances of the same layer type are used.
    /// </para>
    /// </remarks>
    ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix);
}
