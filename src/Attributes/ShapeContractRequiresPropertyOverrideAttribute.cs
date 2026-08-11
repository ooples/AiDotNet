using System;

namespace AiDotNet.Attributes;

/// <summary>
/// Marks a symbolic shape-contract implementation that is available only when a concrete model
/// overrides a named virtual property supplying model-specific shape metadata.
/// </summary>
/// <remarks>
/// This keeps conditional base-class contracts declarative and statically discoverable. Conformance
/// tooling can reject an inherited default before constructing a paper-scale model, while a derived
/// model that supplies the required metadata is automatically treated as a concrete contract without
/// overriding <c>OutputAxesFor</c> itself.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true, Inherited = false)]
public sealed class ShapeContractRequiresPropertyOverrideAttribute : Attribute
{
    /// <summary>Gets the virtual property that a concrete model must override.</summary>
    public string PropertyName { get; }

    /// <summary>Gets the explanation reported by metadata consumers.</summary>
    public string Reason { get; }

    /// <summary>Creates a conditional shape-contract declaration.</summary>
    /// <param name="propertyName">Name of a virtual property declared beside the contract.</param>
    /// <param name="reason">A non-empty explanation of why the property is required.</param>
    public ShapeContractRequiresPropertyOverrideAttribute(string propertyName, string reason)
    {
        if (string.IsNullOrWhiteSpace(propertyName))
            throw new ArgumentException("A shape-contract requirement must name its property.", nameof(propertyName));
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("A shape-contract requirement must explain why it is needed.", nameof(reason));

        PropertyName = propertyName;
        Reason = reason;
    }
}
