namespace AiDotNet.Attributes;

/// <summary>
/// Marks mathematical model parameters that become available through <c>Fit</c> rather than a
/// gradient step. The generator keeps the slot in the manifest before fitting and derives all
/// count, read, write and restore behavior after materialization.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class FittedParameterAttribute : Attribute
{
    /// <summary>Gets the lifecycle for fitted state.</summary>
    public AiDotNet.Models.Parameters.ParameterAvailability Availability { get; set; }
        = AiDotNet.Models.Parameters.ParameterAvailability.Fit;
}
