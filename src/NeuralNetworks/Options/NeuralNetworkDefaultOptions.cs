using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the NeuralNetwork.
/// </summary>
public class NeuralNetworkDefaultOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets explicit hidden-layer widths for the default stack, input side first; null sizes the
    /// hidden layers from the architecture's complexity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used only when the architecture supplies no layers of its own. Left null, the network keeps its
    /// established sizing: one, two or three hidden layers by <c>NetworkComplexity</c>, widths derived
    /// from the input size.
    /// </para>
    /// <para><b>For Beginners:</b> <c>new[] { 64, 32 }</c> means two hidden layers, the first with 64
    /// neurons and the second with 32.</para>
    /// </remarks>
    public int[]? HiddenLayers { get; set; }
}
