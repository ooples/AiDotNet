using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the DomainDecompositionPINN.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This model splits a large domain into pieces, trains a small network on
/// each, and then makes the pieces agree where they meet. The interface weights control how
/// strictly that agreement is enforced.
/// </para>
/// </remarks>
public class DomainDecompositionPINNOptions : PhysicsInformedOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public DomainDecompositionPINNOptions()
    {
        NumCollocationPointsPerSubdomain = 5000;
        PdeWeight = 1.0;
        BoundaryWeight = 1.0;
        InterfaceWeight = 10.0;
        InterfaceGradientWeight = 1.0;
        SchwarzIterations = 1;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public DomainDecompositionPINNOptions(DomainDecompositionPINNOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        NumCollocationPointsPerSubdomain = other.NumCollocationPointsPerSubdomain;
        PdeWeight = other.PdeWeight;
        BoundaryWeight = other.BoundaryWeight;
        InterfaceWeight = other.InterfaceWeight;
        InterfaceGradientWeight = other.InterfaceGradientWeight;
        SchwarzIterations = other.SchwarzIterations;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets how many interior points each subdomain evaluates the PDE at. Default: 5000.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Named per-subdomain rather than NumCollocationPoints because that is what it counts here,
    /// and because the total scales with the number of subdomains.
    /// </para>
    /// </remarks>
    public int NumCollocationPointsPerSubdomain { get; set; }

    /// <summary>
    /// Gets or sets the weight on the PDE residual loss term. Default: 1.0.
    /// </summary>
    public double PdeWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the boundary condition loss term. Default: 1.0.
    /// </summary>
    public double BoundaryWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on matching values across subdomain interfaces. Default: 10.0.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Weighted heavily on purpose — if the pieces disagree where they
    /// meet, the assembled solution is discontinuous and wrong however good each piece is.</para>
    /// </remarks>
    public double InterfaceWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on matching gradients across subdomain interfaces. Default: 1.0.
    /// </summary>
    public double InterfaceGradientWeight { get; set; }

    /// <summary>
    /// Gets or sets the number of Schwarz alternating iterations. Default: 1.
    /// </summary>
    public int SchwarzIterations { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    /// <remarks>
    /// <para>Weights may legitimately be zero; the two counts may not.</para>
    /// </remarks>
    public void Validate()
    {
        Require(NumCollocationPointsPerSubdomain, nameof(NumCollocationPointsPerSubdomain));
        Require(SchwarzIterations, nameof(SchwarzIterations));
    }
}
