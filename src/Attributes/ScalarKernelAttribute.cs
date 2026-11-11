namespace AiDotNet.Attributes;

/// <summary>
/// Marks a member whose element-by-element tensor access is deliberate, exempting it from the ADNELEM001/ADNELEM002
/// element-wise gate.
/// </summary>
/// <remarks>
/// <para>
/// The gate flags a <c>Tensor&lt;T&gt;</c>, <c>Vector&lt;T&gt;</c> or <c>Matrix&lt;T&gt;</c> indexer, or a per-element
/// <c>INumericOperations&lt;T&gt;</c> call, inside a loop: work an engine op or a span loop does many times faster. A
/// member that must stay scalar, such as a reference implementation a test compares against or a loop over a handful
/// of elements, says so here with the reason.
/// </para>
/// <para><b>For Beginners:</b> put this on a method only when reading tensor values one at a time is intended, and
/// say why in <see cref="Reason"/>.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor | AttributeTargets.Property | AttributeTargets.Class
    | AttributeTargets.Struct, Inherited = false)]
public sealed class ScalarKernelAttribute : Attribute
{
    /// <summary>Creates the exemption.</summary>
    /// <param name="reason">Why this member's per-element access is intended.</param>
    public ScalarKernelAttribute(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("State why the per-element access is intended.", nameof(reason));
        Reason = reason;
    }

    /// <summary>Why this member's per-element access is intended.</summary>
    public string Reason { get; }
}
