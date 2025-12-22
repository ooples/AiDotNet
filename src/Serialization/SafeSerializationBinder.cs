using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.Serialization;

/// <summary>
/// A custom serialization binder that restricts deserialization to safe types within the AiDotNet namespace.
/// </summary>
/// <remarks>
/// <para>
/// This binder helps prevent deserialization attacks by only allowing types from the AiDotNet namespace
/// and common .NET framework types to be deserialized. This is important when using TypeNameHandling.Auto
/// (or other TypeNameHandling modes) with Newtonsoft.Json.
/// </para>
/// <para><b>For Beginners:</b> When loading a saved model from a file, we need to know what types of objects
/// to create. However, if an attacker crafts a malicious file, they might try to trick the system into
/// creating dangerous objects. This binder acts as a security guard, only allowing known-safe types.
/// </para>
/// <para><b>Security Note:</b> Always prefer TypeNameHandling.Auto over TypeNameHandling.All as it minimizes
/// type information exposure while still supporting polymorphic deserialization.</para>
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
    {
        "AiDotNet.",
        "System.Collections.Generic.",
    };

    /// <summary>
    /// Exact allowed type full names.
    /// </summary>
    /// <remarks>
    /// Note: System.Object is intentionally NOT included in this list.
    /// While System.Object itself is harmless, allowing it could enable polymorphic
    /// deserialization attacks where object-typed properties are deserialized to
    /// dangerous concrete types. The recursive validation in <see cref="IsAllowedType"/>
    /// validates actual concrete types, but excluding System.Object provides defense in depth.
    /// </remarks>
    private static readonly HashSet<string> AllowedTypeFullNames = new HashSet<string>(StringComparer.Ordinal)
    {
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
        // Note: System.Object is intentionally excluded for security reasons
    };

    /// <summary>
    /// Gets the type to deserialize given the serialized type name.
    /// </summary>
    /// <param name="assemblyName">The assembly name from the serialized data.</param>
    /// <param name="typeName">The type name from the serialized data.</param>
    /// <returns>The resolved type, or throws if the type is not allowed.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the type is not allowed for deserialization.</exception>
    public Type BindToType(string? assemblyName, string typeName)
    {
        // Resolve the type first using the default binder
        var resolvedType = _defaultBinder.BindToType(assemblyName, typeName);

        // Then validate the resolved type recursively
        if (!IsAllowedType(resolvedType))
        {
            throw new InvalidOperationException(
                $"Type '{typeName}' resolved to '{resolvedType?.FullName}' is not allowed for deserialization. " +
                "Only types from the AiDotNet namespace and basic .NET types are permitted.");
        }

        return resolvedType;
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
    /// Checks if a resolved type is allowed for deserialization.
    /// This method recursively validates generic type arguments, array elements, and nullable types.
    /// </summary>
    /// <param name="type">The resolved type to check.</param>
    /// <returns>True if the type and all its components are allowed, false otherwise.</returns>
    private static bool IsAllowedType(Type? type)
    {
        if (type == null)
        {
            return false;
        }

        // Reject dangerous type categories
        if (type.IsPointer || type.IsByRef)
        {
            return false;
        }

        // Allow generic type definitions if they are in allowed namespaces.
        // Note: Generic type definitions *do* contain generic parameters; rejecting them would block all closed generics too,
        // because we validate the generic type definition as part of the closed generic validation path.
        if (type.IsGenericTypeDefinition)
        {
            var defNamespace = type.Namespace ?? "";
            if (defNamespace.Length == 0)
            {
                return false;
            }

            return AllowedNamespacePrefixes.Any(prefix =>
                (defNamespace + ".").StartsWith(prefix, StringComparison.Ordinal));
        }

        // Reject partially-open generic types (unbound type parameters)
        if (type.ContainsGenericParameters)
        {
            return false;
        }

        // Handle array types - recursively check element type
        if (type.IsArray)
        {
            var elementType = type.GetElementType();
            return IsAllowedType(elementType);
        }

        // Handle nullable value types - recursively check underlying type
        var underlyingNullable = Nullable.GetUnderlyingType(type);
        if (underlyingNullable != null)
        {
            return IsAllowedType(underlyingNullable);
        }

        // Check exact type full names (primitives and safe types)
        var fullName = type.FullName ?? "";
        if (AllowedTypeFullNames.Contains(fullName))
        {
            return true;
        }

        // Check namespace prefixes
        var typeNamespace = type.Namespace ?? "";
        if (typeNamespace.Length > 0)
        {
            bool namespaceAllowed = AllowedNamespacePrefixes.Any(prefix =>
                (typeNamespace + ".").StartsWith(prefix, StringComparison.Ordinal));

            if (namespaceAllowed)
            {
                // For non-generic types in allowed namespaces, allow directly
                if (!type.IsGenericType)
                {
                    return true;
                }

                // For generic types, validate the generic type definition is also allowed
                var genericTypeDef = type.GetGenericTypeDefinition();
                if (!IsAllowedType(genericTypeDef))
                {
                    return false;
                }

                // Recursively validate ALL generic type arguments
                var genericArgs = type.GetGenericArguments();
                return genericArgs.All(IsAllowedType);
            }
        }

        return false;
    }
}
