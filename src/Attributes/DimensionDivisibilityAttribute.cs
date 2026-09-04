namespace AiDotNet.Attributes;

/// <summary>
/// Declares that one integer model dimension must be evenly divisible by another
/// integer configuration property.
/// </summary>
/// <remarks>
/// The relationship is part of the architecture, not merely input validation. Test-scale
/// option generation reads this metadata after reducing paper-scale values so it can preserve
/// constraints such as <c>HiddenDim % NumAttentionHeads == 0</c> without per-model branches.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class DimensionDivisibilityAttribute : Attribute
{
    /// <summary>Initializes a new divisibility relationship.</summary>
    /// <param name="dimensionProperty">The integer dimension that must be divisible.</param>
    /// <param name="divisorProperty">The integer property that divides the dimension.</param>
    public DimensionDivisibilityAttribute(string dimensionProperty, string divisorProperty)
    {
        if (string.IsNullOrWhiteSpace(dimensionProperty))
            throw new ArgumentException("A dimension property name is required.", nameof(dimensionProperty));
        if (string.IsNullOrWhiteSpace(divisorProperty))
            throw new ArgumentException("A divisor property name is required.", nameof(divisorProperty));

        DimensionProperty = dimensionProperty;
        DivisorProperty = divisorProperty;
    }

    /// <summary>Gets the integer dimension property name.</summary>
    public string DimensionProperty { get; }

    /// <summary>Gets the integer divisor property name.</summary>
    public string DivisorProperty { get; }
}
