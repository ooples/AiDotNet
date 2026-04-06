using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SpikingNeuralNetwork.
/// </summary>
public class SpikingNeuralNetworkOptions : NeuralNetworkOptions
{
    private double _readoutLearningRate = 0.005;
    private int _stdpWindow = 20;

    /// <summary>
    /// Learning rate for STDP weight updates. Balances convergence speed vs stability.
    /// Must be positive.
    /// </summary>
    public double ReadoutLearningRate
    {
        get => _readoutLearningRate;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), value, "ReadoutLearningRate must be positive.");
            _readoutLearningRate = value;
        }
    }

    /// <summary>
    /// STDP time window (number of time steps to consider for spike-timing correlations).
    /// Larger windows capture longer-range temporal dependencies but increase computation.
    /// Must be at least 1.
    /// </summary>
    public int StdpWindow
    {
        get => _stdpWindow;
        set
        {
            if (value < 1)
                throw new ArgumentOutOfRangeException(nameof(value), value, "StdpWindow must be at least 1.");
            _stdpWindow = value;
        }
    }
}
