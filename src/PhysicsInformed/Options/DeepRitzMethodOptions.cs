using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the DeepRitzMethod.
/// </summary>
public class DeepRitzMethodOptions : PhysicsInformedOptions
{

    /// <summary>
    /// Gets or sets num quadrature points. Default: <c>10000</c>.
    /// </summary>
    public int NumQuadraturePoints { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the boundary penalty beta on mean |u - g|^2 over the boundary data. Default: <c>500</c>,
    /// the value E and Yu (2018) use for their Poisson examples (Sec. 3.1).
    /// </summary>
    public double BoundaryPenalty { get; set; } = 500.0;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(NumQuadraturePoints, nameof(NumQuadraturePoints));
        if (BoundaryPenalty < 0 || double.IsNaN(BoundaryPenalty))
        {
            throw new ArgumentOutOfRangeException(nameof(BoundaryPenalty), "Boundary penalty must be non-negative.");
        }
    }
}
