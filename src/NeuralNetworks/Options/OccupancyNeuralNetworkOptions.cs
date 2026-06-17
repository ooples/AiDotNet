using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the OccupancyNeuralNetwork.
/// </summary>
public class OccupancyNeuralNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initial learning rate for the default Adam (AMSGrad) optimizer the network
    /// builds when no optimizer is supplied. Default 1e-4. The occupancy head is a
    /// sigmoid; at the previous 1e-2 rate a short memorization run drove the
    /// pre-sigmoid logit straight into hard saturation (σ → exactly 1.0 for every
    /// query within a few steps), so two distinct points produced bit-identical
    /// probabilities and the network looked collapsed (#1208/#1221-style uniform
    /// output). The lower rate keeps the logit in the sigmoid's responsive range, so
    /// distinct inputs stay distinguishable, while loss still decreases steadily on
    /// the memorization task. Tune for larger or noisier occupancy datasets.
    /// </summary>
    public double BaseOptimizerInitialLearningRate { get; set; } = 0.0001;
}
