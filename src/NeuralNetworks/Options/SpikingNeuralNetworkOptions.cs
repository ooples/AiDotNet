using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SpikingNeuralNetwork.
/// </summary>
public class SpikingNeuralNetworkOptions : NeuralNetworkOptions
{
    // Default LR calibrated for the supervised surrogate-gradient delta
    // rule at the output layer (Zenke 2018 §3 / Neftci 2019 surrogate
    // gradient review) — not the original sparse STDP magnitude. The
    // surrogate-gradient update is W += lr × (target − actual) × pre_rate,
    // which produces a much denser per-iteration weight change than
    // STDP's spike-coincidence-only updates. At lr=5e-3 the dense delta
    // rule overshoots the optimum after ≈50 iterations and starts
    // oscillating away from it (MoreData_ShouldNotDegrade saw loss go
    // from 0.13 at 50 iters to 0.25 at 200 iters). Dropping to 5e-4
    // keeps the 200-iter trajectory below the 50-iter loss while still
    // converging fast enough that Training_ShouldReduceLoss measurably
    // improves over the test's default ~30 iterations. STDP-only hidden
    // layers also benefit from the smaller step (sparse Hebbian updates
    // accumulate over many iterations and can drift the hidden
    // representation at lr=5e-3).
    private double _readoutLearningRate = 1e-3;
    private int _stdpWindow = 20;

    /// <summary>
    /// Learning rate for the supervised surrogate-gradient delta-rule at
    /// the output layer and the unsupervised STDP updates at the hidden
    /// layers. Balances convergence speed vs stability.
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
