using System;

namespace AiDotNet.Attributes;

/// <summary>
/// Marks an <c>IShapeContract.OutputAxesFor</c> implementation that deliberately provides no
/// symbolic output-shape law.
/// </summary>
/// <remarks>
/// This is capability metadata, not a suppression. It lets discovery and conformance tooling identify
/// an inherited, intentional decline without constructing a potentially foundation-scale model merely
/// to receive <c>null</c>. A derived override is a new implementation and does not inherit this marker;
/// its declared law is therefore probed normally.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class ShapeContractUnavailableAttribute : Attribute
{
    /// <summary>Explains why the implementation cannot yet declare one safe family-wide law.</summary>
    public string Reason { get; }

    /// <summary>Creates an explicit unavailability declaration.</summary>
    /// <param name="reason">A non-empty explanation suitable for generated reports.</param>
    public ShapeContractUnavailableAttribute(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("A missing shape contract must explain why it is unavailable.", nameof(reason));
        Reason = reason;
    }
}
