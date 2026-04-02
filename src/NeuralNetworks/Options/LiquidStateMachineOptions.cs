using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the LiquidStateMachine.
/// </summary>
public class LiquidStateMachineOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Learning rate for the readout layer. Per Maass et al. 2002, the reservoir is fixed
    /// and only the readout is trained. A low LR prevents overfitting/divergence since
    /// the readout is a simple linear mapping.
    /// </summary>
    public double ReadoutLearningRate { get; set; } = 0.0001;
}
