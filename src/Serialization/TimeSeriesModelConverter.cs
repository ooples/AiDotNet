namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter that handles all types derived from TimeSeriesModelBase.
/// </summary>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of any model type that inherits from 
/// TimeSeriesModelBase by leveraging the binary serialization methods they all inherit.
/// </para>
/// <para><b>For Beginners:</b> This converter makes sure all time series models can be properly
/// saved and loaded, including their private fields. It works for all model types without
/// needing a separate converter for each one.
/// </para>
/// </remarks>
public class TimeSeriesModelConverter : JsonConverter
{
    // Cache of model base types to avoid repeated reflection
    private static readonly Dictionary<Type, bool> _isTimeSeriesModelCache = [];

    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        // Check cache first
        if (_isTimeSeriesModelCache.TryGetValue(objectType, out bool result))
            return result;

        // Check if type inherits from TimeSeriesModelBase<T> for any T
        bool isTimeSeriesModel = IsTimeSeriesModelType(objectType);

        // Cache the result
        _isTimeSeriesModelCache[objectType] = isTimeSeriesModel;

        return isTimeSeriesModel;
    }

    /// <summary>
    /// Checks if a type is derived from TimeSeriesModelBase.
    /// </summary>
    private bool IsTimeSeriesModelType(Type objectType)
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

                // Check if this is a TimeSeriesModelBase
                if (fullName.Contains("TimeSeriesModelBase"))
                    return true;
            }

            // Check base type
            currentType = currentType.BaseType;
        }

        return false;
    }

    /// <summary>
    /// Writes the JSON representation of the TimeSeriesModel object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The TimeSeriesModel to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value == null)
        {
            writer.WriteNull();
            return;
        }

        try
        {
            // Create a container for our serialized data
            JObject container = new JObject();

            // Store the concrete type information
            Type concreteType = value.GetType();
            container["$type"] = concreteType.AssemblyQualifiedName;

            // First attempt: Use Serialize method (high-level API)
            MethodInfo? serializeMethod = concreteType.GetMethod("Serialize");
            if (serializeMethod != null)
            {
                try
                {
                    var binaryData = (byte[]?)serializeMethod.Invoke(value, null);
                    container["BinaryData"] = Convert.ToBase64String(binaryData ?? []);

                    // Skip serializing public properties to avoid circular references
                    container.WriteTo(writer);
                    return;
                }
                catch (Exception ex)
                {
                    // Log the error if possible
                    container["SerializeError"] = ex.Message;
                    // If high-level serialization fails, continue to low-level approach
                }
            }

            // Second attempt: Use SerializeCore method (low-level API)
            MethodInfo? serializeCoreMethod = FindMethod(concreteType, "SerializeCore",
                new[] { typeof(BinaryWriter) });

            if (serializeCoreMethod != null)
            {
                try
                {
                    using MemoryStream ms = new MemoryStream();
                    using BinaryWriter bw = new BinaryWriter(ms);

                    // Invoke SerializeCore
                    serializeCoreMethod.Invoke(value, new object[] { bw });

                    // Get the binary data
                    byte[] binaryData = ms.ToArray();
                    container["BinaryData"] = Convert.ToBase64String(binaryData);

                    // Skip serializing public properties to avoid circular references
                    container.WriteTo(writer);
                    return;
                }
                catch (Exception ex)
                {
                    // Log the error if possible
                    container["SerializeCoreError"] = ex.Message;
                }
            }

            // If both binary methods fail, create a simpler representation with just essential info
            container["ModelType"] = concreteType.Name;
            container.WriteTo(writer);
        }
        catch (Exception ex)
        {
            // Create a minimal error object
            JObject errorObject = new JObject();
            errorObject["$type"] = value.GetType().AssemblyQualifiedName;
            errorObject["SerializationError"] = $"Failed to serialize: {ex.Message}";
            errorObject.WriteTo(writer);
        }
    }

    /// <summary>
    /// Reads the JSON representation of the TimeSeriesModel object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The TimeSeriesModel object deserialized from JSON.</returns>
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
                throw new JsonSerializationException("Missing type information for TimeSeriesModel deserialization.");
            }

            // Resolve the concrete type
            Type? concreteType = ResolveType(typeString ?? string.Empty);
            if (concreteType == null)
            {
                throw new JsonSerializationException($"Could not resolve type: {typeString}");
            }

            // Check if we have binary data
            string? binaryString = container["BinaryData"]?.Value<string>();
            if (string.IsNullOrEmpty(binaryString))
            {
                // No binary data, try standard deserialization if there's PublicData
                JToken? publicData = container["PublicData"];
                if (publicData != null)
                {
                    return publicData.ToObject(concreteType, serializer);
                }

                // Or just create an empty instance
                return CreateModelInstance(concreteType);
            }

            // Create a new instance of the model
            object? model = CreateModelInstance(concreteType);
            if (model == null)
            {
                throw new JsonSerializationException($"Failed to create instance of type {concreteType.Name}");
            }

            // Get the binary data
            byte[] binaryData = Convert.FromBase64String(binaryString);

            // First attempt: Use Deserialize method (high-level API)
            MethodInfo? deserializeMethod = concreteType.GetMethod("Deserialize");
            if (deserializeMethod != null)
            {
                try
                {
                    deserializeMethod.Invoke(model, new object[] { binaryData });
                    return model;
                }
                catch
                {
                    // If high-level deserialization fails, continue to low-level approach
                }
            }

            // Second attempt: Use DeserializeCore method (low-level API)
            MethodInfo? deserializeCoreMethod = FindMethod(concreteType, "DeserializeCore",
                new[] { typeof(BinaryReader) });

            if (deserializeCoreMethod != null)
            {
                using MemoryStream ms = new MemoryStream(binaryData);
                using BinaryReader br = new BinaryReader(ms);

                // Invoke DeserializeCore
                deserializeCoreMethod.Invoke(model, new object[] { br });

                // Make sure model is marked as trained and parameters are synced
                SetModelAsTrained(model);

                return model;
            }

            // If both methods fail, return the basic instance
            return model;
        }
        catch (Exception ex)
        {
            throw new JsonSerializationException($"Error deserializing TimeSeriesModel: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Resolves a type from its string representation, checking all loaded assemblies if needed.
    /// </summary>
    private Type? ResolveType(string typeString)
    {
        // First try direct resolution
        Type? type = Type.GetType(typeString);
        if (type != null)
            return type;

        // Try to find it in all loaded assemblies
        string simpleTypeName = typeString.Split(',')[0].Trim();
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
    /// Creates a new instance of a TimeSeriesModel type, handling any constructor requirements.
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

    /// <summary>
    /// Finds a method on a type using reflection, including non-public methods.
    /// </summary>
    private MethodInfo? FindMethod(Type type, string methodName, Type[] parameterTypes)
    {
        return type.GetMethod(methodName,
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
            null, parameterTypes, null);
    }

    /// <summary>
    /// Sets a model as trained and ensures its parameters are synchronized.
    /// </summary>
    private void SetModelAsTrained(object model)
    {
        Type modelType = model.GetType();

        // Set IsTrained property
        PropertyInfo? isTrainedProperty = modelType.GetProperty("IsTrained",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (isTrainedProperty != null)
        {
            isTrainedProperty.SetValue(model, true);
        }

        // Try to sync ModelParameters with GetCurrentState
        MethodInfo? getCurrentStateMethod = FindMethod(modelType, "GetCurrentState", Type.EmptyTypes);
        PropertyInfo? modelParamsProperty = modelType.GetProperty("ModelParameters",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

        if (getCurrentStateMethod != null && modelParamsProperty != null)
        {
            try
            {
                object? state = getCurrentStateMethod.Invoke(model, null);
                if (state != null)
                {
                    modelParamsProperty.SetValue(model, state);
                }
            }
            catch
            {
                // Ignore errors in parameter synchronization
            }
        }
    }
}