namespace AiDotNet.Attributes;

/// <summary>
/// Marks a field or property as another view of parameter storage owned by a different member.
/// </summary>
/// <remarks>
/// Alias members are intentionally excluded from generated parameter storage so the same values
/// cannot be counted, optimized, or restored more than once. <see cref="Target"/> names the stable
/// owning member for diagnostics and manifest tooling.
/// </remarks>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class ParameterAliasAttribute : Attribute
{
    public ParameterAliasAttribute(string target)
    {
        if (string.IsNullOrWhiteSpace(target))
            throw new ArgumentException("An alias target is required.", nameof(target));

        Target = target;
    }

    /// <summary>Gets the name of the member that owns the parameter storage.</summary>
    public string Target { get; }
}
