namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter for interface types with built-in diagnostics to debug serialization issues.
/// </summary>
/// <remarks>
/// <para>
/// This enhanced converter handles the serialization and deserialization of interfaces with detailed diagnostics
/// to identify why interfaces might be coming back as null.
/// </para>
/// </remarks>
public class DiagnosticInterfaceConverter : JsonConverter
{
    private static bool _debugMode = true;
    private static readonly string _logFilePath = "interface_converter_log.txt";

    private void Log(string message)
    {
        if (_debugMode)
        {
            try
            {
                File.AppendAllText(_logFilePath, $"[{DateTime.Now}] {message}{Environment.NewLine}");
                Debug.WriteLine($"InterfaceConverter: {message}");
            }
            catch
            {
                // Ignore logging errors - shouldn't affect normal operation
            }
        }
    }

    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        bool isApplicable = objectType.IsInterface || (objectType.IsAbstract && !objectType.IsSealed);
        Log($"CanConvert({objectType.FullName}) => {isApplicable}");
        return isApplicable;
    }

    /// <summary>
    /// Writes the JSON representation of the interface object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The interface object to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        Log($"WriteJson called with value: {value?.GetType().FullName ?? "null"}");

        if (value == null)
        {
            Log("Writing null value");
            writer.WriteNull();
            return;
        }

        // Get the concrete type information
        Type concreteType = value.GetType();

        try
        {
            // Create a container with both type and object data
            JObject container = new JObject();
            container["$type"] = $"InterfaceContainer, AiDotNet"; // Marker to identify our container
            container["ConcreteTypeName"] = concreteType.AssemblyQualifiedName;

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
                    if (converter.GetType() != typeof(DiagnosticInterfaceConverter))
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
            container["AssemblyName"] = concreteType.Assembly.GetName().Name;
            container["TypeFullName"] = concreteType.FullName;

            // Add type details for better diagnostics
            Log($"Serializing type: {concreteType.FullName} from assembly: {concreteType.Assembly.GetName().Name}");

            // Write the container to JSON
            container.WriteTo(writer);
            Log("Successfully wrote interface container to JSON");
        }
        catch (Exception ex)
        {
            Log($"Error in WriteJson: {ex.Message}\n{ex.StackTrace}");

            // Fallback - try to serialize directly
            try
            {
                JObject directObject = JObject.FromObject(value);
                directObject["$typeName"] = concreteType.AssemblyQualifiedName;
                directObject.WriteTo(writer);
                Log("Used fallback direct serialization");
            }
            catch (Exception fallbackEx)
            {
                Log($"Fallback serialization failed: {fallbackEx.Message}");
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
        Log($"ReadJson called for type: {objectType.FullName}");

        if (reader.TokenType == JsonToken.Null)
        {
            Log("Encountered null token, returning null");
            return null;
        }

        try
        {
            // Clone reader position before loading JObject
            JObject container = JObject.Load(reader);
            Log($"Loaded JSON object with properties: {string.Join(", ", container.Properties().Select(p => p.Name))}");

            // Check if this is our container format
            bool isContainer = container["$type"]?.ToString() == "InterfaceContainer, AiDotNet";

            string? typeName;
            JToken? objectData;

            if (isContainer)
            {
                Log("Detected interface container format");
                // Get type from our container
                typeName = container["ConcreteTypeName"]?.ToString();
                objectData = container["ObjectData"];

                // Log additional diagnostic info
                Log($"Container Assembly: {container["AssemblyName"]}");
                Log($"Container TypeName: {container["TypeFullName"]}");
            }
            else
            {
                Log("Using standard type resolution");
                // Use standard type tokens
                typeName = container["$type"]?.ToString() ?? container["$typeName"]?.ToString();
                objectData = container;
            }

            Log($"Resolved type name: {typeName ?? "null"}");

            if (string.IsNullOrEmpty(typeName))
            {
                Log("Type name is empty, returning null");
                return null;
            }

            // Try to find the type
            Type? concreteType = null;

            try
            {
                // First, try direct lookup
                concreteType = Type.GetType(typeName);
                Log($"Direct Type.GetType lookup: {(concreteType != null ? "success" : "failed")}");

                // If that fails, try to find it in loaded assemblies
                if (concreteType == null && typeName != null)
                {
                    Log("Searching in loaded assemblies");
                    string typeNameWithoutAssembly = typeName.Split(',')[0].Trim();

                    foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
                    {
                        try
                        {
                            // Skip dynamic assemblies
                            if (assembly.IsDynamic)
                                continue;

                            Log($"Checking assembly: {assembly.GetName().Name}");
                            foreach (Type type in assembly.GetTypes())
                            {
                                if (type.FullName == typeNameWithoutAssembly)
                                {
                                    concreteType = type;
                                    Log($"Found type in assembly: {assembly.GetName().Name}");
                                    break;
                                }
                            }

                            if (concreteType != null)
                                break;
                        }
                        catch (Exception)
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
                        Log($"Trying to load assembly: {assemblyName}");
                        try
                        {
                            Assembly assembly = Assembly.Load(assemblyName);
                            concreteType = assembly.GetType(typeFullName);
                            Log($"Loaded type from assembly directly: {(concreteType != null ? "success" : "failed")}");
                        }
                        catch (Exception ex)
                        {
                            Log($"Failed to load assembly: {ex.Message}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"Error resolving type: {ex.Message}");
            }

            if (concreteType == null)
            {
                Log($"Failed to resolve type: {typeName}");
                return null;
            }

            // Validate type compatibility
            if (!objectType.IsAssignableFrom(concreteType))
            {
                Log($"Type {concreteType.FullName} is not assignable to {objectType.FullName}");
                return null;
            }

            Log($"Successfully resolved compatible type: {concreteType.FullName}");

            // Create instance and populate
            try
            {
                // First try using the serializer's capabilities
                object? result = null;

                if (isContainer && objectData != null)
                {
                    Log("Deserializing from ObjectData property");
                    result = objectData.ToObject(concreteType, serializer);
                }
                else
                {
                    Log("Deserializing directly from container");
                    result = container.ToObject(concreteType, serializer);
                }

                if (result != null)
                {
                    Log($"Successfully deserialized object of type: {result.GetType().FullName}");
                    return result;
                }

                // If that fails, try creating an instance and populating it manually
                Log("Trying manual instantiation and population");
                object instance = Activator.CreateInstance(concreteType)
                    ?? throw new InvalidOperationException($"Failed to create instance of {concreteType.FullName}");

                if (objectData != null)
                {
                    serializer.Populate(objectData.CreateReader(), instance);
                    Log("Populated instance via serializer.Populate");
                }

                return instance;
            }
            catch (Exception ex)
            {
                Log($"Error deserializing object: {ex.Message}\n{ex.StackTrace}");

                // Last resort - try just creating an instance without populating
                try
                {
                    Log("Last resort: creating unpopulated instance");
                    return Activator.CreateInstance(concreteType);
                }
                catch
                {
                    Log("Could not create instance even as last resort");
                    return null;
                }
            }
        }
        catch (Exception ex)
        {
            Log($"Unhandled exception in ReadJson: {ex.Message}\n{ex.StackTrace}");
            return null;
        }
    }
}