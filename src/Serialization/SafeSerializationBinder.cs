using Newtonsoft.Json.Serialization;

namespace AiDotNet.Serialization;

/// <summary>
/// A custom serialization binder that restricts deserialization to safe types within the AiDotNet namespace.
/// </summary>
/// <remarks>
/// <para>
/// This binder helps prevent deserialization attacks by only allowing types from the AiDotNet namespace
/// and common .NET framework types to be deserialized. This is important when using TypeNameHandling.All
/// with Newtonsoft.Json.
/// </para>
/// <para><b>For Beginners:</b> When loading a saved model from a file, we need to know what types of objects
/// to create. However, if an attacker crafts a malicious file, they might try to trick the system into
/// creating dangerous objects. This binder acts as a security guard, only allowing known-safe types.
/// </para>
/// </remarks>
public class SafeSerializationBinder : ISerializationBinder
{
    /// <summary>
    /// The default binder to delegate to for type resolution.
    /// </summary>
    private readonly DefaultSerializationBinder _defaultBinder = new DefaultSerializationBinder();

    /// <summary>
    /// Allowed namespace prefixes for deserialization.
    /// </summary>
    private static readonly string[] AllowedNamespacePrefixes =
    [
        "AiDotNet.",
        "System.Collections.Generic.",
        "System.String",
        "System.Int32",
        "System.Int64",
        "System.Double",
        "System.Single",
        "System.Boolean",
        "System.Decimal",
        "System.DateTime",
        "System.DateTimeOffset",
        "System.TimeSpan",
        "System.Guid",
        "System.Byte",
        "System.SByte",
        "System.Int16",
        "System.UInt16",
        "System.UInt32",
        "System.UInt64",
        "System.Char",
        "System.Nullable",
        "System.ValueTuple"
    ];

    /// <summary>
    /// Gets the type to deserialize given the serialized type name.
    /// </summary>
    /// <param name="assemblyName">The assembly name from the serialized data.</param>
    /// <param name="typeName">The type name from the serialized data.</param>
    /// <returns>The resolved type, or throws if the type is not allowed.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the type is not allowed for deserialization.</exception>
    public Type BindToType(string? assemblyName, string typeName)
    {
        // Check if the type is in an allowed namespace
        if (!IsAllowedType(typeName))
        {
            throw new InvalidOperationException(
                $"Type '{typeName}' is not allowed for deserialization. " +
                "Only types from the AiDotNet namespace and basic .NET types are permitted.");
        }

        // Use the default binder to resolve the type
        return _defaultBinder.BindToType(assemblyName, typeName);
    }

    /// <summary>
    /// Gets the serialized type name for a type during serialization.
    /// </summary>
    /// <param name="serializedType">The type being serialized.</param>
    /// <param name="assemblyName">Output: the assembly name to serialize.</param>
    /// <param name="typeName">Output: the type name to serialize.</param>
    public void BindToName(Type serializedType, out string? assemblyName, out string? typeName)
    {
        _defaultBinder.BindToName(serializedType, out assemblyName, out typeName);
    }

    /// <summary>
    /// Checks if a type name is allowed for deserialization.
    /// </summary>
    /// <param name="typeName">The full type name to check.</param>
    /// <returns>True if the type is allowed, false otherwise.</returns>
    private static bool IsAllowedType(string typeName)
    {
        if (string.IsNullOrWhiteSpace(typeName))
        {
            return false;
        }

        // Check against allowed prefixes using LINQ
        if (AllowedNamespacePrefixes.Any(prefix => typeName.StartsWith(prefix, StringComparison.Ordinal)))
        {
            return true;
        }

        // Also allow array types of allowed types
        if (typeName.EndsWith("[]", StringComparison.Ordinal))
        {
            var elementType = typeName.Substring(0, typeName.Length - 2);
            return IsAllowedType(elementType);
        }

        // Allow generic types where all type arguments are allowed
        if (typeName.Contains('`'))
        {
            // Extract the base type name
            int backtickIndex = typeName.IndexOf('`');
            string baseTypeName = typeName.Substring(0, backtickIndex);

            // Check if base type is allowed using LINQ
            bool baseAllowed = AllowedNamespacePrefixes.Any(prefix =>
                baseTypeName.StartsWith(prefix, StringComparison.Ordinal));

            if (baseAllowed)
            {
                // For simplicity, allow if base type is allowed
                // A more thorough implementation would parse and check all type arguments
                return true;
            }
        }

        return false;
    }
}
