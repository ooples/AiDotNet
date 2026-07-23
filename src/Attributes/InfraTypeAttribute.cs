using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the type of infrastructure component (Tier 3 metadata).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to classes that provide platform infrastructure
/// rather than ML functionality. Infrastructure components handle storage, serving, caching,
/// and other operational concerns.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [InfraType(InfraType.Cache)]
/// public class PredictionCache&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class InfraTypeAttribute : Attribute
{
    /// <summary>
    /// Gets the infrastructure type.
    /// </summary>
    public InfraType Type { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InfraTypeAttribute"/> class.
    /// </summary>
    /// <param name="type">The infrastructure type.</param>
    public InfraTypeAttribute(InfraType type)
    {
        Type = type;
    }
}
