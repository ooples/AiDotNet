using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SpikingNeuralNetwork.
/// </summary>
public class SpikingNeuralNetworkOptions : NeuralNetworkOptions
{
    // The composite default network performs full surrogate-gradient BPTT
    // through 20 time steps and reuses each synapse at every step. Its Adam
    // gradient is therefore the sum of all temporal uses, not a single-step
    // delta-rule update. A per-step rate such as 5e-4 is effectively amplified
    // by the unrolled readout integration and overshoots immediately. The 5e-6
    // default is calibrated for that complete temporal gradient; callers using
    // a shorter unroll or a custom topology can set their own rate explicitly.
    private double _readoutLearningRate = 5e-6;
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
    /// STDP time window (number of time steps to consider for spike-timing correlations)
    /// applied by the unsupervised pair-based STDP learning rule (Gerstner &amp; Kistler 2002).
    /// Larger windows capture longer-range temporal dependencies but increase computation.
    /// Must be at least 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Scope:</b> This knob applies ONLY to unsupervised STDP weight updates on the
    /// hidden spiking layers (e.g. when training the reservoir as an unsupervised feature
    /// extractor). The supervised <see cref="SpikingNeuralNetwork{T}.Train"/> path
    /// freezes hidden layers and trains the readout via Zenke 2018 surrogate-gradient
    /// Adam — pair-based STDP is non-supervised and decoupled from a supervised MSE
    /// loss (it can drift the hidden representation in directions that hurt loss, which
    /// is why supervised Train ignores this value). Proper supervised hidden-layer
    /// training requires BPTT-through-time on the surrogate gradient (Zenke 2018 §3.2);
    /// when that lands, supervised Train will start honoring this window. Setting
    /// <c>StdpWindow</c> with <c>SpikingNeuralNetwork.Train</c> alone is a no-op and
    /// should not be expected to change supervised-training behavior.
    /// </para>
    /// </remarks>
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
