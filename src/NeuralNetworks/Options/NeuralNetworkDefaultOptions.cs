using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the NeuralNetwork.
/// </summary>
public class NeuralNetworkDefaultOptions : NeuralNetworkOptions
{
    /// <summary>
    /// The global gradient-norm bound the network clips its training gradients to; 0 turns the network's own clipping
    /// off, leaving only its optimizer's (default 1.0, PyTorch's usual transformer setting).
    /// </summary>
    public double MaxGradNorm { get; set; } = 1.0;
}
