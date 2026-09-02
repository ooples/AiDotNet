namespace AiDotNet.Attributes;

/// <summary>Identifies the architectural role of an integer constructor parameter.</summary>
/// <remarks>
/// This metadata makes relationships between scalar constructor arguments explicit without relying
/// on parameter-name strings. Reflection-based construction and validation can therefore preserve
/// attention geometry even when a parameter is renamed.
/// </remarks>
[AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
public sealed class ModelDimensionRoleAttribute : Attribute
{
    /// <summary>Initializes a new role declaration.</summary>
    public ModelDimensionRoleAttribute(ModelDimensionRole role) => Role = role;

    /// <summary>Gets the declared architectural role.</summary>
    public ModelDimensionRole Role { get; }
}

/// <summary>Architectural roles that participate in cross-parameter dimension constraints.</summary>
public enum ModelDimensionRole
{
    /// <summary>The width split across attention heads.</summary>
    AttentionDimension,

    /// <summary>The number of attention heads that divides the attention dimension.</summary>
    AttentionHeadCount,
}
