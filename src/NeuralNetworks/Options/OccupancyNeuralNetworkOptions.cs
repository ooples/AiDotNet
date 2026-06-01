using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the OccupancyNeuralNetwork.
/// </summary>
public class OccupancyNeuralNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initial learning rate for the default Adam (AMSGrad) optimizer the
    /// network builds when no optimizer is supplied. Default 0.01 — higher than
    /// the framework's 1e-3 default because the Conditional-LayerNorm ResNet
    /// decoder normalizes the pre-output activation, which damps how fast the
    /// occupancy head's bias moves; the larger rate keeps single-pair
    /// memorization within a typical training budget while staying inside the
    /// single-step parameter-stability bound. Tune down for larger or noisier
    /// occupancy datasets.
    /// </summary>
    public double BaseOptimizerInitialLearningRate { get; set; } = 0.01;
}
