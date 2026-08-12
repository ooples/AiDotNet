namespace AiDotNet.Attributes;

/// <summary>
/// Marks mathematical parameters that participate in the parameter surface but are excluded from
/// optimizer updates.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class FrozenParameterAttribute : Attribute
{
    /// <summary>Gets or sets when this fixed parameter becomes available.</summary>
    public AiDotNet.Models.Parameters.ParameterAvailability Availability { get; set; }
        = AiDotNet.Models.Parameters.ParameterAvailability.Construction;
}
