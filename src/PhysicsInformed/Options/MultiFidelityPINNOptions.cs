using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the MultiFidelityPINN.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A multi-fidelity PINN learns from two sources at once: a lot of cheap,
/// rough data and a little expensive, accurate data. The weights here decide how much it trusts
/// each, and how hard it tries to keep the two consistent.
/// </para>
/// </remarks>
public class MultiFidelityPINNOptions : PhysicsInformedOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public MultiFidelityPINNOptions()
    {
        NumCollocationPoints = 10000;
        LowFidelityWeight = 1.0;
        HighFidelityWeight = 10.0;
        CorrelationWeight = 1.0;
        PdeWeight = 1.0;
        BoundaryWeight = 1.0;
        FreezeLowFidelityAfterPretraining = true;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public MultiFidelityPINNOptions(MultiFidelityPINNOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        NumCollocationPoints = other.NumCollocationPoints;
        LowFidelityWeight = other.LowFidelityWeight;
        HighFidelityWeight = other.HighFidelityWeight;
        CorrelationWeight = other.CorrelationWeight;
        PdeWeight = other.PdeWeight;
        BoundaryWeight = other.BoundaryWeight;
        FreezeLowFidelityAfterPretraining = other.FreezeLowFidelityAfterPretraining;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets how many interior points the PDE residual is evaluated at. Default: 10000.
    /// </summary>
    public int NumCollocationPoints { get; set; }

    /// <summary>
    /// Gets or sets the weight on the cheap, plentiful low-fidelity data. Default: 1.0.
    /// </summary>
    public double LowFidelityWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the expensive, scarce high-fidelity data. Default: 10.0,
    /// deliberately ten times the low-fidelity weight because there is far less of it.
    /// </summary>
    public double HighFidelityWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the term tying the two fidelities together. Default: 1.0.
    /// </summary>
    public double CorrelationWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the PDE residual loss term. Default: 1.0.
    /// </summary>
    public double PdeWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the boundary condition loss term. Default: 1.0.
    /// </summary>
    public double BoundaryWeight { get; set; }

    /// <summary>
    /// Gets or sets whether the low-fidelity network is frozen once pretraining ends.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Once the model has learned the rough shape from the cheap data,
    /// freezing that part stops the scarce accurate data from undoing it.</para>
    /// </remarks>
    public bool FreezeLowFidelityAfterPretraining { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    /// <remarks>
    /// <para>Weights may legitimately be zero, so only the collocation count is required.</para>
    /// </remarks>
    public void Validate()
    {
        Require(NumCollocationPoints, nameof(NumCollocationPoints));
    }
}
