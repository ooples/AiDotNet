using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the CapsuleNetwork.
/// </summary>
public class CapsuleNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Learning rate for SGD parameter updates during training. Default: 0.01.
    /// </summary>
    public double LearningRate { get; set; } = 0.01;
}
