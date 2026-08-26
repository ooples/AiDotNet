using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the OctonionNeuralNetwork.
/// </summary>
public class OctonionNeuralNetworkOptions : NeuralNetworkOptions
{
    /// <summary>Initial learning rate used for the paper's ten-epoch ramp-in phase.</summary>
    public double InitialLearningRate { get; set; } = 0.01;

    /// <summary>Nesterov momentum used in the CIFAR experiments.</summary>
    public double Momentum { get; set; } = 0.9;

    /// <summary>Epoch at which the learning rate increases from 0.01 to 0.1.</summary>
    public int RampEpoch { get; set; } = 10;

    /// <summary>Epoch at which the learning rate returns to 0.01.</summary>
    public int FirstDecayEpoch { get; set; } = 100;

    /// <summary>Epoch at which the learning rate decays to 0.001.</summary>
    public int SecondDecayEpoch { get; set; } = 150;

    public OctonionNeuralNetworkOptions()
    {
    }

    public OctonionNeuralNetworkOptions(OctonionNeuralNetworkOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        InitialLearningRate = other.InitialLearningRate;
        Momentum = other.Momentum;
        RampEpoch = other.RampEpoch;
        FirstDecayEpoch = other.FirstDecayEpoch;
        SecondDecayEpoch = other.SecondDecayEpoch;
    }
}
