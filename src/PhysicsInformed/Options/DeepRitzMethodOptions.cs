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
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(NumQuadraturePoints, nameof(NumQuadraturePoints));
    }
}
