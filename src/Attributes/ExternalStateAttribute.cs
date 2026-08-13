namespace AiDotNet.Attributes;

/// <summary>
/// Marks numeric state owned by an external runtime. The generated AiDotNet parameter graph does
/// not count, optimize or restore this member.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class ExternalStateAttribute : Attribute
{
}
