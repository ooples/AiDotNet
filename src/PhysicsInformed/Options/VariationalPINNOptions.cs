using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the VariationalPINN.
/// </summary>
public class VariationalPINNOptions : PhysicsInformedOptions
{

    /// <summary>
    /// Gets or sets num quadrature points. Default: <c>10000</c>.
    /// </summary>
    public int NumQuadraturePoints { get; set; } = 10000;

    /// <summary>
    /// Gets or sets num test functions. Default: <c>10</c>.
    /// </summary>
    public int NumTestFunctions { get; set; } = 10;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(NumQuadraturePoints, nameof(NumQuadraturePoints));
        Require(NumTestFunctions, nameof(NumTestFunctions));
    }
}
