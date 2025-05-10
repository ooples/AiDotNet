using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using AiDotNet.Regression;

namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter that handles all types derived from RegressionModelBase.
/// </summary>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of any model type that inherits from 
/// RegressionModelBase by leveraging the binary serialization methods they all inherit.
/// </para>
/// <para><b>For Beginners:</b> This converter makes sure all regression models can be properly
/// saved and loaded, including their private fields. It works for all model types without
/// needing a separate converter for each one.
/// </para>
/// </remarks>
public class RegressionModelConverter : JsonConverter
{
    // Cache of model base types to avoid repeated reflection
    private static readonly Dictionary<Type, bool> _isRegressionModelCache = new Dictionary<Type, bool>();

    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        // Check cache first
        if (_isRegressionModelCache.TryGetValue(objectType, out bool result))
            return result;

        // Check if type inherits from RegressionModelBase<T> for any T
        bool isRegressionModel = IsRegressionModelType(objectType);

        // Cache the result
        _isRegressionModelCache[objectType] = isRegressionModel;

        return isRegressionModel;
    }

    /// <summary>
    /// Checks if a type is derived from RegressionModelBase.
    /// </summary>
    private bool IsRegressionModelType(Type objectType)
    {
        if (objectType == null)
            return false;

        // Check all base types
        Type? currentType = objectType;
        while (currentType != null && currentType != typeof(object))
        {
            // Check if this is a generic type
            if (currentType.IsGenericType)
            {
                Type genericTypeDef = currentType.GetGenericTypeDefinition();
                string fullName = genericTypeDef.FullName ?? string.Empty;

                // Check if this is a RegressionModelBase
                if (fullName.Contains("RegressionModelBase"))
                    return true;
            }

            // Check base type
            currentType = currentType.BaseType;
        }

        return false;
    }

    /// <summary>
    /// Writes the JSON representation of the RegressionModel object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The RegressionModel to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value == null)
        {
            writer.WriteNull();
            return;
        }

        JObject container = new JObject();

        // Store the concrete type information
        Type concreteType = value.GetType();
        container["$type"] = concreteType.AssemblyQualifiedName;

        // Get binary data through the Serialize method that all regression models inherit
        try
        {
            // Try to get the Serialize method from the model
            MethodInfo? serializeMethod = concreteType.GetMethod("Serialize");

            if (serializeMethod != null)
            {
                // Invoke the Serialize method to get binary data
                var binaryData = (byte[]?)serializeMethod.Invoke(value, null);

                if (binaryData != null && binaryData.Length > 0)
                {
                    // Store the binary data as base64 string
                    container["BinaryData"] = Convert.ToBase64String(binaryData);

                    // Write the container to JSON
                    container.WriteTo(writer);
                    return;
                }
            }
        }
        catch (Exception ex)
        {
            container["SerializationError"] = ex.Message;
        }

        // Fallback: basic serialization of public properties
        try
        {
            // Create a container for public properties
            var properties = new JObject();

            // Use reflection to get public properties
            foreach (var prop in concreteType.GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                if (prop.CanRead)
                {
                    try
                    {
                        object? propValue = prop.GetValue(value);
                        properties[prop.Name] = propValue != null ? JToken.FromObject(propValue, serializer) : null;
                    }
                    catch
                    {
                        // Skip properties that cause serialization issues
                    }
                }
            }

            container["PublicData"] = properties;
            container.WriteTo(writer);
        }
        catch (Exception ex)
        {
            // Last resort: just write the error
            JObject errorObject = new JObject
            {
                ["$type"] = concreteType.AssemblyQualifiedName,
                ["SerializationError"] = $"Failed to serialize: {ex.Message}"
            };
            errorObject.WriteTo(writer);
        }
    }

    /// <summary>
    /// Reads the JSON representation of the RegressionModel object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The RegressionModel object deserialized from JSON.</returns>
    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null)
            return null;

        try
        {
            // Read the JSON container
            JObject container = JObject.Load(reader);

            // Get the concrete type information
            string? typeString = container["$type"]?.Value<string>();
            if (string.IsNullOrEmpty(typeString))
            {
                throw new JsonSerializationException("Missing type information for RegressionModel deserialization.");
            }

            // Resolve the concrete type
            Type? concreteType = ResolveType(typeString);
            if (concreteType == null)
            {
                throw new JsonSerializationException($"Could not resolve type: {typeString}");
            }

            // Try to create a new instance of the model
            object? model = CreateModelInstance(concreteType);
            if (model == null)
            {
                throw new JsonSerializationException($"Failed to create instance of type {concreteType.Name}");
            }

            // Check if we have binary data
            string? binaryString = container["BinaryData"]?.Value<string>();
            if (!string.IsNullOrEmpty(binaryString))
            {
                // Get the binary data
                byte[] binaryData = Convert.FromBase64String(binaryString);

                // Try to find and invoke the Deserialize method
                MethodInfo? deserializeMethod = concreteType.GetMethod("Deserialize");
                if (deserializeMethod != null)
                {
                    try
                    {
                        deserializeMethod.Invoke(model, new object[] { binaryData });
                        return model;
                    }
                    catch (Exception ex)
                    {
                        throw new JsonSerializationException($"Failed to deserialize binary data: {ex.Message}", ex);
                    }
                }
                else
                {
                    throw new JsonSerializationException($"Type {concreteType.Name} does not have a Deserialize method.");
                }
            }
            else if (container["PublicData"] != null)
            {
                // Fallback: deserialize from public properties
                var publicData = container["PublicData"] as JObject;
                serializer.Populate(publicData?.CreateReader() ?? container.CreateReader(), model);
                return model;
            }
            else
            {
                throw new JsonSerializationException("No binary data or public data found for deserialization.");
            }
        }
        catch (Exception ex)
        {
            throw new JsonSerializationException($"Error deserializing RegressionModel: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Resolves a type from its string representation, checking all loaded assemblies if needed.
    /// </summary>
    private Type? ResolveType(string? typeString)
    {
        if (string.IsNullOrEmpty(typeString))
            return null;

        // First try direct resolution
        Type? type = Type.GetType(typeString);
        if (type != null)
            return type;

        // Try to find it in all loaded assemblies
        string simpleTypeName = typeString?.Split(',')[0].Trim() ?? string.Empty;
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic)
                continue;

            try
            {
                type = assembly.GetType(simpleTypeName);
                if (type != null)
                    return type;
            }
            catch
            {
                // Skip assemblies that can't be examined
            }
        }

        return null;
    }

    /// <summary>
    /// Creates a new instance of a RegressionModel type, handling any constructor requirements.
    /// </summary>
    private object? CreateModelInstance(Type modelType)
    {
        try
        {
            // First attempt: Default parameterless constructor
            return Activator.CreateInstance(modelType);
        }
        catch
        {
            // Second attempt: Look for constructors with optional parameters
            var constructors = modelType.GetConstructors();
            foreach (var ctor in constructors)
            {
                var parameters = ctor.GetParameters();
                if (parameters.Length == 0)
                    continue;

                // Check if all parameters are optional
                bool allOptional = true;
                foreach (var param in parameters)
                {
                    if (!param.IsOptional)
                    {
                        allOptional = false;
                        break;
                    }
                }

                if (allOptional)
                {
                    // Create an array of default values
                    object?[] args = new object[parameters.Length];
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        args[i] = Type.Missing;
                    }

                    return ctor.Invoke(args);
                }
            }

            return null;
        }
    }
}