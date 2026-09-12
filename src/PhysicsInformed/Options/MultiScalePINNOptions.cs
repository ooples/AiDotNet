using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the Multi-Scale PINN model.
/// </summary>
public class MultiScalePINNOptions : PhysicsInformedOptions
{

    /// <summary>
    /// Gets or sets num collocation points per scale. Default: <c>5000</c>.
    /// </summary>
    public int NumCollocationPointsPerScale { get; set; } = 5000;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(NumCollocationPointsPerScale, nameof(NumCollocationPointsPerScale));
    }
}
