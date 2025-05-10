namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter for interface types to enable proper serialization and deserialization.
/// </summary>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of interfaces by storing the concrete
/// type information along with the object data.
/// </para>
/// </remarks>
public class InterfaceJsonConverter : JsonConverter
{
    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        return objectType.IsInterface || (objectType.IsAbstract && !objectType.IsSealed);
    }

    /// <summary>
    /// Writes the JSON representation of the interface object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The interface object to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value == null)
        {
            writer.WriteNull();
            return;
        }

        // Get the concrete type information
        Type concreteType = value.GetType();

        try
        {
            // Create a container with both type and object data
            JObject container = new JObject();
            container["$type"] = "InterfaceContainer, AiDotNet"; // Marker to identify our container
            container["ConcreteTypeName"] = concreteType.AssemblyQualifiedName;
            container["AssemblyName"] = concreteType.Assembly.GetName().Name;
            container["TypeFullName"] = concreteType.FullName;

            // Temporarily disable TypeNameHandling to avoid recursive type information
            JsonSerializerSettings clonedSettings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.None
            };

            // Clone the converters but exclude this one to avoid recursion
            if (serializer.Converters != null)
            {
                foreach (var converter in serializer.Converters)
                {
                    if (converter.GetType() != typeof(InterfaceJsonConverter))
                    {
                        clonedSettings.Converters.Add(converter);
                    }
                }
            }

            // Create new serializer with our settings
            JsonSerializer nestedSerializer = JsonSerializer.Create(clonedSettings);

            // Serialize the object to a JObject
            JToken? objectData;
            using (JTokenWriter tokenWriter = new JTokenWriter())
            {
                nestedSerializer.Serialize(tokenWriter, value);
                objectData = tokenWriter.Token;
            }

            container["ObjectData"] = objectData;

            // Write the container to JSON
            container.WriteTo(writer);
        }
        catch (Exception)
        {
            // Fallback - try to serialize directly
            try
            {
                JObject directObject = JObject.FromObject(value);
                directObject["$typeName"] = concreteType.AssemblyQualifiedName;
                directObject.WriteTo(writer);
            }
            catch
            {
                writer.WriteNull();
            }
        }
    }

    /// <summary>
    /// Reads the JSON representation of the interface object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The interface object deserialized from JSON.</returns>
    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null)
        {
            return null;
        }

        try
        {
            // Clone reader position before loading JObject
            JObject container = JObject.Load(reader);

            // Check if this is our container format
            bool isContainer = container["$type"]?.ToString() == "InterfaceContainer, AiDotNet";

            string? typeName;
            JToken? objectData;

            if (isContainer)
            {
                // Get type from our container
                typeName = container["ConcreteTypeName"]?.ToString();
                objectData = container["ObjectData"];
            }
            else
            {
                // Use standard type tokens
                typeName = container["$type"]?.ToString() ?? container["$typeName"]?.ToString();
                objectData = container;
            }

            if (string.IsNullOrEmpty(typeName))
            {
                return null;
            }

            // Try to find the type
            Type? concreteType = null;

            try
            {
                // First, try direct lookup
                concreteType = Type.GetType(typeName);

                // If that fails, try to find it in loaded assemblies
                if (concreteType == null && typeName != null)
                {
                    string typeNameWithoutAssembly = typeName.Split(',')[0].Trim();

                    foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
                    {
                        try
                        {
                            // Skip dynamic assemblies
                            if (assembly.IsDynamic)
                                continue;

                            foreach (Type type in assembly.GetTypes())
                            {
                                if (type.FullName == typeNameWithoutAssembly)
                                {
                                    concreteType = type;
                                    break;
                                }
                            }

                            if (concreteType != null)
                                break;
                        }
                        catch
                        {
                            // Skip assemblies that can't be inspected
                        }
                    }
                }

                // If we still can't find it, try loading the assembly explicitly
                if (concreteType == null && isContainer)
                {
                    string? assemblyName = container["AssemblyName"]?.ToString();
                    string? typeFullName = container["TypeFullName"]?.ToString();

                    if (!string.IsNullOrEmpty(assemblyName) && !string.IsNullOrEmpty(typeFullName))
                    {
                        try
                        {
                            Assembly assembly = Assembly.Load(assemblyName);
                            concreteType = assembly.GetType(typeFullName);
                        }
                        catch
                        {
                            // Failed to load assembly
                        }
                    }
                }
            }
            catch
            {
                // Error resolving type
            }

            if (concreteType == null)
            {
                return null;
            }

            // Validate type compatibility
            if (!objectType.IsAssignableFrom(concreteType))
            {
                return null;
            }

            // Create instance and populate
            try
            {
                // First try using the serializer's capabilities
                object? result = null;

                if (isContainer && objectData != null)
                {
                    result = objectData.ToObject(concreteType, serializer);
                }
                else
                {
                    result = container.ToObject(concreteType, serializer);
                }

                if (result != null)
                {
                    return result;
                }

                // If that fails, try creating an instance and populating it manually
                object instance = Activator.CreateInstance(concreteType)
                    ?? throw new InvalidOperationException($"Failed to create instance of {concreteType.FullName}");

                if (objectData != null)
                {
                    serializer.Populate(objectData.CreateReader(), instance);
                }

                return instance;
            }
            catch
            {
                // Last resort - try just creating an instance without populating
                try
                {
                    return Activator.CreateInstance(concreteType);
                }
                catch
                {
                    return null;
                }
            }
        }
        catch
        {
            // Unhandled exception in ReadJson
            return null;
        }
    }
}