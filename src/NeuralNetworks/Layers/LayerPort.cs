namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Declares a named input or output port on a layer.
/// Ports enable multi-input layers (e.g., DiffusionResBlock needs "input" + "time_embed")
/// and provide compile-time documentation of a layer's data contract.
/// </summary>
/// <param name="Name">Port name (e.g., "input", "time_embed", "query", "key", "value").</param>
/// <param name="Shape">Expected tensor shape for this port.</param>
/// <param name="Required">If true, Forward throws when this port is missing. Default: true.</param>
/// <remarks>
/// <para><b>For Beginners:</b> A port is like a labeled plug on the layer.
/// Just as a TV has separate ports for HDMI, USB, and power, a neural network layer
/// can have separate ports for different types of input data.</para>
/// </remarks>
public sealed record LayerPort(string Name, int[] Shape, bool Required = true);
