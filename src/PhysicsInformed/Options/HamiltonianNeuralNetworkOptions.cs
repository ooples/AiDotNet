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
        if (other is null)
            throw new ArgumentNullException(nameof(other));
            EncoderLayerCount = other.EncoderLayerCount;
        HiddenLayerCount = other.HiddenLayerCount;
        HiddenDimension = other.HiddenDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        SymplecticSolverMaxIterations = other.SymplecticSolverMaxIterations;
        SymplecticSolverTolerance = other.SymplecticSolverTolerance;
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

    /// <summary>Gets or sets the most fixed-point iterations of the implicit momentum update in Simulate.</summary>
    /// <value>The iteration cap. The default is 100.</value>
    /// <remarks>
    /// <para>Symplectic Euler for a general (non-separable) Hamiltonian evaluates the force at the new momentum,
    /// <c>p_new = p - dt * dH/dq(q, p_new)</c>, which is solved by fixed-point iteration. A separable Hamiltonian
    /// converges after one correction.</para>
    /// <para><b>For Beginners:</b> Each simulation step refines its momentum estimate until it stops changing; this caps
    /// how many refinements one step may take.</para>
    /// </remarks>
    public int SymplecticSolverMaxIterations { get; set; } = 100;

    /// <summary>Gets or sets the convergence tolerance of the implicit momentum update in Simulate.</summary>
    /// <value>The relative tolerance. The default is 1e-10.</value>
    /// <remarks>
    /// <para>Iteration stops when the largest momentum change is at most <c>tolerance * (1 + max |p|)</c>, or when it
    /// no longer decreases (the precision of the numeric type is reached).</para>
    /// <para><b>For Beginners:</b> Smaller values solve each step more exactly at the cost of more network
    /// evaluations.</para>
    /// </remarks>
    public double SymplecticSolverTolerance { get; set; } = 1e-10;

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
        if (SymplecticSolverMaxIterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(SymplecticSolverMaxIterations), "SymplecticSolverMaxIterations must be positive.");
        if (double.IsNaN(SymplecticSolverTolerance) || double.IsInfinity(SymplecticSolverTolerance) || SymplecticSolverTolerance <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(SymplecticSolverTolerance), "SymplecticSolverTolerance must be finite and positive.");
    }
}
