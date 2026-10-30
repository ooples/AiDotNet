using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the FeedForwardNeuralNetwork.
/// </summary>
public class FeedForwardNeuralNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the width of each hidden layer of the default stack, input side first.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used only when the architecture supplies no layers of its own. Each entry becomes a ReLU dense
    /// layer; the output layer comes from the architecture. The default, two layers of 64, is the stack
    /// this network has always built.
    /// </para>
    /// <para><b>For Beginners:</b> <c>new[] { 64, 32 }</c> means two hidden layers, the first with 64
    /// neurons and the second with 32.</para>
    /// </remarks>
    public int[] HiddenLayers { get; set; } = { 64, 64 };
}
