using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the PhysicsInformedNeuralNetwork.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A PINN trains on three things at once — any data you give it, the
/// differential equation it must satisfy, and the conditions at the edges of the domain. The
/// weights here decide how much each of those matters. Tuning them is often the trickiest part of
/// training a PINN.
/// </para>
/// </remarks>
public class PhysicsInformedNeuralNetworkOptions : PhysicsInformedOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public PhysicsInformedNeuralNetworkOptions()
    {
        NumCollocationPoints = 10000;
        DataWeight = 1.0;
        PdeWeight = 1.0;
        BoundaryWeight = 1.0;
        InitialWeight = 1.0;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public PhysicsInformedNeuralNetworkOptions(PhysicsInformedNeuralNetworkOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        NumCollocationPoints = other.NumCollocationPoints;
        DataWeight = other.DataWeight;
        PdeWeight = other.PdeWeight;
        BoundaryWeight = other.BoundaryWeight;
        InitialWeight = other.InitialWeight;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets how many interior points the PDE residual is evaluated at. Default: 10000.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model checks the physics at this many scattered points
    /// inside the domain. More points means a better-enforced equation and slower training.</para>
    /// </remarks>
    public int NumCollocationPoints { get; set; }

    /// <summary>
    /// Gets or sets the weight on the data-fitting loss term. Default: 1.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Previously a nullable constructor parameter defaulting to null, which
    /// <c>PhysicsInformedLoss</c> resolved as <c>dataWeight ?? 1.0</c>. Non-nullable with the same
    /// 1.0 is exactly equivalent and does not ask the reader to trace the null through.
    /// </para>
    /// </remarks>
    public double DataWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the PDE residual loss term. Default: 1.0.
    /// </summary>
    public double PdeWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the boundary condition loss term. Default: 1.0.
    /// </summary>
    public double BoundaryWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight on the initial condition loss term. Default: 1.0.
    /// </summary>
    public double InitialWeight { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    /// <remarks>
    /// <para>
    /// Only the collocation count is required. Every weight may legitimately be zero — that is how
    /// a term is switched off — so requiring them positive would reject valid configurations.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        Require(NumCollocationPoints, nameof(NumCollocationPoints));
    }
}
