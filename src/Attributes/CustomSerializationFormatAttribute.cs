namespace AiDotNet.Attributes;

/// <summary>
/// Marks a type whose public checkpoint format has semantics that cannot be represented by the
/// ordinary generated state envelope.
/// </summary>
/// <remarks>
/// This is intentionally an explicit, type-level opt-in. It is reserved for established wire
/// formats that require compatibility framing, validation, or a representation different from the
/// in-memory state. Ordinary model and layer state must continue to use generated declarations.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class CustomSerializationFormatAttribute : Attribute
{
}
