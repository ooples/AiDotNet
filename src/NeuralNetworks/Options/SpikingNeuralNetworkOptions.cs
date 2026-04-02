using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SpikingNeuralNetwork.
/// </summary>
public class SpikingNeuralNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Learning rate for STDP weight updates. Balances convergence speed vs stability.
    /// </summary>
    public double ReadoutLearningRate { get; set; } = 0.005;

    /// <summary>
    /// STDP time window (number of time steps to consider for spike-timing correlations).
    /// Larger windows capture longer-range temporal dependencies but increase computation.
    /// </summary>
    public int StdpWindow { get; set; } = 20;
}
