using System.Collections;
using System.Reflection;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Generates JSON Schema (the dialect chat models consume for tool parameters and structured output)
/// from .NET types and method parameters using reflection.
/// </summary>
/// <remarks>
/// <para>
/// This is what lets a plain C# method become a model-callable tool without hand-writing a schema.
/// Primitive types map to their JSON counterparts, enums become string enumerations, collections become
/// arrays, string-keyed dictionaries become open objects, and other classes become nested objects built
/// from their public readable properties (with a depth/cycle guard so recursive types terminate).
/// </para>
/// <para><b>For Beginners:</b> A model needs a description of a tool's inputs written in "JSON Schema".
/// Writing that by hand is tedious and error-prone. This class reads your C# types and writes the schema
/// for you — e.g. a method taking <c>(string city, int days = 3)</c> becomes an object with a required
/// string <c>city</c> and an optional integer <c>days</c>.
/// </para>
/// </remarks>
public static class JsonSchemaGenerator
{
    private const int MaxDepth = 6;

    /// <summary>
    /// Builds an object schema (<c>type: object</c> with <c>properties</c> and <c>required</c>) describing
    /// a method's parameters.
    /// </summary>
    /// <param name="parameters">The parameters to describe.</param>
    /// <returns>A JSON Schema object suitable for use as a tool's parameter schema.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is <c>null</c>.</exception>
    public static JObject ForParameters(IReadOnlyList<ParameterInfo> parameters)
    {
        Guard.NotNull(parameters);

        var properties = new JObject();
        var required = new JArray();

        foreach (var parameter in parameters)
        {
            // Skip cancellation tokens — they are supplied by the runtime, not the model.
            if (parameter.ParameterType == typeof(CancellationToken))
            {
                continue;
            }

            var name = parameter.Name ?? "arg";
            var underlying = Nullable.GetUnderlyingType(parameter.ParameterType) ?? parameter.ParameterType;
            var schema = ForType(underlying, new HashSet<Type>(), 0);

            var attribute = parameter.GetCustomAttribute<ToolParameterAttribute>();
            if (attribute is not null && !string.IsNullOrWhiteSpace(attribute.Description))
            {
                schema["description"] = attribute.Description;
            }

            properties[name] = schema;

            if (IsRequired(parameter, attribute))
            {
                required.Add(name);
            }
        }

        var result = new JObject
        {
            ["type"] = "object",
            ["properties"] = properties
        };

        if (required.Count > 0)
        {
            result["required"] = required;
        }

        return result;
    }

    /// <summary>
    /// Builds a JSON Schema fragment describing a single .NET type.
    /// </summary>
    /// <param name="type">The type to describe.</param>
    /// <returns>A JSON Schema object for the type.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="type"/> is <c>null</c>.</exception>
    public static JObject ForType(Type type)
    {
        Guard.NotNull(type);
        var underlying = Nullable.GetUnderlyingType(type) ?? type;
        return ForType(underlying, new HashSet<Type>(), 0);
    }

    private static bool IsRequired(ParameterInfo parameter, ToolParameterAttribute? attribute)
    {
        if (attribute?.Required is { } explicitRequired)
        {
            return explicitRequired;
        }

        // Optional / defaulted parameters and Nullable<T> value types are not required.
        if (parameter.IsOptional || parameter.HasDefaultValue)
        {
            return false;
        }

        return Nullable.GetUnderlyingType(parameter.ParameterType) is null;
    }

    private static JObject ForType(Type type, HashSet<Type> visiting, int depth)
    {
        if (type.IsEnum)
        {
            var names = new JArray();
            foreach (var name in Enum.GetNames(type))
            {
                names.Add(name);
            }

            return new JObject { ["type"] = "string", ["enum"] = names };
        }

        if (type == typeof(Guid))
        {
            return new JObject { ["type"] = "string", ["format"] = "uuid" };
        }

        if (type == typeof(DateTime) || type == typeof(DateTimeOffset))
        {
            return new JObject { ["type"] = "string", ["format"] = "date-time" };
        }

        if (type == typeof(TimeSpan))
        {
            return new JObject { ["type"] = "string", ["format"] = "duration" };
        }

        switch (Type.GetTypeCode(type))
        {
            case TypeCode.Boolean:
                return new JObject { ["type"] = "boolean" };
            case TypeCode.Byte:
            case TypeCode.SByte:
            case TypeCode.Int16:
            case TypeCode.UInt16:
            case TypeCode.Int32:
            case TypeCode.UInt32:
            case TypeCode.Int64:
            case TypeCode.UInt64:
                return new JObject { ["type"] = "integer" };
            case TypeCode.Single:
            case TypeCode.Double:
            case TypeCode.Decimal:
                return new JObject { ["type"] = "number" };
            case TypeCode.Char:
            case TypeCode.String:
                return new JObject { ["type"] = "string" };
        }

        if (TryGetDictionaryValueType(type, out var valueType))
        {
            return new JObject
            {
                ["type"] = "object",
                ["additionalProperties"] = depth >= MaxDepth
                    ? new JObject()
                    : ForType(Nullable.GetUnderlyingType(valueType) ?? valueType, visiting, depth + 1)
            };
        }

        if (TryGetEnumerableElementType(type, out var elementType))
        {
            return new JObject
            {
                ["type"] = "array",
                ["items"] = depth >= MaxDepth
                    ? new JObject()
                    : ForType(Nullable.GetUnderlyingType(elementType) ?? elementType, visiting, depth + 1)
            };
        }

        // Complex object: expand public readable instance properties, guarding against cycles/depth.
        if (depth >= MaxDepth || !visiting.Add(type))
        {
            return new JObject { ["type"] = "object" };
        }

        var properties = new JObject();
        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (!property.CanRead || property.GetIndexParameters().Length > 0)
            {
                continue;
            }

            var propertyType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
            properties[property.Name] = ForType(propertyType, visiting, depth + 1);
        }

        visiting.Remove(type);
        return new JObject { ["type"] = "object", ["properties"] = properties };
    }

    private static bool TryGetDictionaryValueType(Type type, out Type valueType)
    {
        foreach (var candidate in EnumerateTypeAndInterfaces(type))
        {
            if (candidate.IsGenericType && candidate.GetGenericTypeDefinition() == typeof(IDictionary<,>))
            {
                var args = candidate.GetGenericArguments();
                if (args[0] == typeof(string))
                {
                    valueType = args[1];
                    return true;
                }
            }
        }

        valueType = typeof(object);
        return false;
    }

    private static bool TryGetEnumerableElementType(Type type, out Type elementType)
    {
        if (type.IsArray)
        {
            elementType = type.GetElementType() ?? typeof(object);
            return true;
        }

        // Strings are IEnumerable<char> but were already handled as a scalar string above.
        if (typeof(IEnumerable).IsAssignableFrom(type))
        {
            foreach (var candidate in EnumerateTypeAndInterfaces(type))
            {
                if (candidate.IsGenericType && candidate.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                {
                    elementType = candidate.GetGenericArguments()[0];
                    return true;
                }
            }

            elementType = typeof(object);
            return true;
        }

        elementType = typeof(object);
        return false;
    }

    private static IEnumerable<Type> EnumerateTypeAndInterfaces(Type type)
    {
        yield return type;
        foreach (var i in type.GetInterfaces())
        {
            yield return i;
        }
    }
}
