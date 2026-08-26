using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the HamiltonianNeuralNetwork.
/// </summary>
public class HamiltonianNeuralNetworkOptions : PhysicsInformedOptions
{
    /// <summary>Initializes the paper-default HNN options.</summary>
    public HamiltonianNeuralNetworkOptions()
    {
    }

    /// <summary>Initializes a value copy of an existing HNN configuration.</summary>
    /// <param name="other">The configuration to copy.</param>
    public HamiltonianNeuralNetworkOptions(HamiltonianNeuralNetworkOptions other)
    {
        ArgumentNullException.ThrowIfNull(other);
        EncoderLayerCount = other.EncoderLayerCount;
        HiddenLayerCount = other.HiddenLayerCount;
        HiddenDimension = other.HiddenDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        DerivativeStep = other.DerivativeStep;
    }

    /// <summary>Gets or sets the number of tanh hidden layers.</summary>
    /// <value>The number of hidden layers. The default is 2.</value>
    /// <remarks>
    /// The reference implementation for Greydanus et al. (2019) uses two hidden affine/tanh
    /// transformations followed by the Hamiltonian projection.
    /// </remarks>
    public int HiddenLayerCount { get; set; } = 2;

    /// <summary>Gets or sets the width of every hidden layer.</summary>
    /// <value>The hidden width. The default is 200.</value>
    /// <remarks>
    /// The paper reports 200 hidden units for the coordinate-based experiments.
    /// </remarks>
    public int HiddenDimension { get; set; } = 200;

    /// <summary>Gets or sets the fixed Adam learning rate.</summary>
    /// <value>The learning rate. The default is 0.001.</value>
    /// <remarks>
    /// Section 3 of Greydanus et al. (2019) reports Adam with a learning rate of 10^-3.
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the coupled L2 penalty used by Adam.</summary>
    /// <value>The L2 strength. The default is 0.0001.</value>
    /// <remarks>
    /// Appendix A reports weight decay of 10^-4 for the first three coordinate tasks. This is
    /// ordinary Adam weight decay, not decoupled AdamW decay.
    /// </remarks>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>Gets or sets the central-difference displacement used for input derivatives.</summary>
    /// <value>The positive displacement. The default is 0.001.</value>
    /// <remarks>
    /// AiDotNet's first-order tape does not retain the higher-order graph required to differentiate
    /// an input gradient with respect to network parameters. A centered input difference preserves
    /// the paper's symplectic derivative objective while remaining differentiable with respect to
    /// every model parameter. The default is large enough to remain resolvable in single precision.
    /// </remarks>
    public double DerivativeStep { get; set; } = 1e-3;

    /// <summary>Validates the paper training recipe and architecture.</summary>
    public void Validate()
    {
        if (HiddenLayerCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenLayerCount), "HiddenLayerCount must be positive.");
        if (HiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenDimension), "HiddenDimension must be positive.");
        if (double.IsNaN(LearningRate) || double.IsInfinity(LearningRate) || LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), "LearningRate must be finite and positive.");
        if (double.IsNaN(WeightDecay) || double.IsInfinity(WeightDecay) || WeightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(WeightDecay), "WeightDecay must be finite and nonnegative.");
        if (double.IsNaN(DerivativeStep) || double.IsInfinity(DerivativeStep) || DerivativeStep <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(DerivativeStep), "DerivativeStep must be finite and positive.");
    }
}
