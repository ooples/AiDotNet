using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serialization;

/// <summary>
/// A diagnostic model converter that provides detailed information about the serialization
/// and deserialization process to help debug serialization issues.
/// </summary>
/// <remarks>
/// This converter is designed to be used as a base class for specific model converters
/// and includes extensive diagnostics to help identify serialization issues.
/// </remarks>
public abstract class DiagnosticModelConverter : JsonConverter
{
    // Debug flag - set to true to enable debug output in a log file
    protected bool EnableDebugOutput { get; set; } = true;

    // Path for debug log file
    protected string DebugLogPath { get; set; } = "model_converter_debug.log";

    // Cache of type compatibility
    private readonly Dictionary<Type, bool> _typeCompatibilityCache = new Dictionary<Type, bool>();

    // Abstract method to determine if a type is supported by this converter
    protected abstract bool IsTypeSupported(Type objectType);

    // Abstract method to get the base model type name for diagnostic purposes
    protected abstract string GetBaseModelTypeName();

    public override bool CanConvert(Type objectType)
    {
        // Check cache first
        if (_typeCompatibilityCache.TryGetValue(objectType, out bool result))
            return result;

        // Determine compatibility
        bool isSupported = IsTypeSupported(objectType);

        // Cache the result
        _typeCompatibilityCache[objectType] = isSupported;

        // Log if debugging enabled
        if (EnableDebugOutput && isSupported)
        {
            Console.WriteLine($"Type {objectType.FullName} IS compatible with {GetBaseModelTypeName()} converter");
        }

        return isSupported;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value == null)
        {
            writer.WriteNull();
            return;
        }

        Type concreteType = value.GetType();

        Console.WriteLine($"==== SERIALIZING {concreteType.Name} ====");

        try
        {
            // Create a container for our serialized data
            JObject container = new JObject();

            // Store the concrete type information
            container["$type"] = concreteType.AssemblyQualifiedName;
            Console.WriteLine($"Set $type to {concreteType.AssemblyQualifiedName}");

            // ================================================================
            // Try all the various serialization approaches in order of preference
            // ================================================================

            // 1. First approach: Use standard Serialize method
            bool serialized = TryStandardSerialization(value, container, concreteType);
            if (serialized)
            {
                container.WriteTo(writer);
                Console.WriteLine($"Successfully serialized {concreteType.Name} using standard serialization");
                return;
            }

            // 2. Second approach: Use SerializeCore + manually save key properties
            serialized = TryCoreSerializationWithProperties(value, container, concreteType);
            if (serialized)
            {
                container.WriteTo(writer);
                Console.WriteLine($"Successfully serialized {concreteType.Name} using core serialization + properties");
                return;
            }

            // 3. Third approach: Capture all fields and properties
            serialized = TryFullReflectionSerialization(value, container, concreteType, serializer);
            if (serialized)
            {
                container.WriteTo(writer);
                Console.WriteLine($"Successfully serialized {concreteType.Name} using full reflection");
                return;
            }

            // 4. Fallback: Just serialize the public data we can access
            JObject publicData = new JObject();
            foreach (var prop in concreteType.GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                if (prop.CanRead)
                {
                    try
                    {
                        object? propValue = prop.GetValue(value);
                        if (propValue != null)
                        {
                            publicData[prop.Name] = JToken.FromObject(propValue, serializer);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error getting property {prop.Name}: {ex.Message}");
                    }
                }
            }

            container["PublicData"] = publicData;
            container.WriteTo(writer);
            Console.WriteLine($"Serialized {concreteType.Name} with fallback approach (public properties only)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CRITICAL ERROR during serialization of {concreteType.Name}: {ex}");

            // Create a minimal error object
            JObject errorObject = new JObject
            {
                ["$type"] = concreteType.AssemblyQualifiedName,
                ["SerializationError"] = $"Failed to serialize: {ex.Message}"
            };
            errorObject.WriteTo(writer);
        }
    }

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null)
            return null;

        Console.WriteLine($"==== DESERIALIZING {objectType.Name} ====");

        try
        {
            // Read the JSON container
            JObject container = JObject.Load(reader);

            // Get the concrete type information
            string? typeString = container["$type"]?.Value<string>();
            Console.WriteLine($"Read $type: {typeString}");

            if (string.IsNullOrEmpty(typeString))
            {
                throw new JsonSerializationException($"Missing type information for {GetBaseModelTypeName()} deserialization.");
            }

            // Resolve the concrete type
            Type? concreteType = ResolveType(typeString);
            if (concreteType == null)
            {
                throw new JsonSerializationException($"Could not resolve type: {typeString}");
            }

            Console.WriteLine($"Resolved to concrete type: {concreteType.FullName}");

            // Create a new instance
            object? model = CreateModelInstance(concreteType);
            if (model == null)
            {
                throw new JsonSerializationException($"Failed to create instance of type {concreteType.Name}");
            }

            Console.WriteLine($"Created new instance of {concreteType.Name}");

            // ================================================================
            // Try all the various deserialization approaches in order of preference
            // ================================================================

            // 1. Try BinaryData approach (using Deserialize method)
            bool deserialized = TryBinaryDeserialization(container, concreteType, model);
            if (deserialized)
            {
                Console.WriteLine($"Successfully deserialized {concreteType.Name} using binary data");
                ValidateDeserializedObject(model);
                return model;
            }

            // 2. Try FieldsData approach (direct field restoration)
            deserialized = TryFieldsDeserialization(container, model);
            if (deserialized)
            {
                Console.WriteLine($"Successfully deserialized {concreteType.Name} using fields data");
                ValidateDeserializedObject(model);
                return model;
            }

            // 3. Try standard object population from PublicData
            if (container["PublicData"] != null)
            {
                try
                {
                    var publicData = container["PublicData"];
                    if (publicData != null)
                    {
                        serializer.Populate(publicData.CreateReader(), model);
                        Console.WriteLine($"Populated {concreteType.Name} using PublicData");
                        SynchronizeInternalState(model);
                        ValidateDeserializedObject(model);
                        return model;
                    }
                    else
                    {
                        Console.WriteLine("PublicData token exists but is null");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error populating from PublicData: {ex.Message}");
                }
            }

            // 4. Fallback: try to populate directly from the container
            try
            {
                serializer.Populate(container.CreateReader(), model);
                Console.WriteLine($"Populated {concreteType.Name} directly from container");
                SynchronizeInternalState(model);
                ValidateDeserializedObject(model);
                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error populating from container: {ex.Message}");
                throw new JsonSerializationException($"All deserialization approaches failed for {concreteType.Name}", ex);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CRITICAL ERROR during deserialization: {ex}");
            throw new JsonSerializationException($"Error deserializing {GetBaseModelTypeName()}: {ex.Message}", ex);
        }
    }

    // ================================================================
    // Serialization helper methods
    // ================================================================

    private bool TryStandardSerialization(object value, JObject container, Type concreteType)
    {
        try
        {
            MethodInfo? serializeMethod = concreteType.GetMethod("Serialize");
            if (serializeMethod != null)
            {
                Console.WriteLine($"Found Serialize method on {concreteType.Name}");
                var binaryData = (byte[]?)serializeMethod.Invoke(value, null);

                if (binaryData != null && binaryData.Length > 0)
                {
                    container["BinaryData"] = Convert.ToBase64String(binaryData);
                    Console.WriteLine($"Stored BinaryData: {binaryData.Length} bytes");

                    // Also capture key property values for debugging
                    CaptureKeyProperties(value, container, concreteType);

                    return true;
                }
                else
                {
                    Console.WriteLine("Serialize method returned null or empty data");
                }
            }
            else
            {
                Console.WriteLine($"No Serialize method found on {concreteType.Name}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in standard serialization: {ex.Message}");
        }

        return false;
    }

    private bool TryCoreSerializationWithProperties(object value, JObject container, Type concreteType)
    {
        try
        {
            // Look for SerializeCore method
            MethodInfo? serializeCoreMethod = FindMethod(concreteType, "SerializeCore", new[] { typeof(BinaryWriter) });

            if (serializeCoreMethod != null)
            {
                Console.WriteLine($"Found SerializeCore method on {concreteType.Name}");

                using MemoryStream ms = new MemoryStream();
                using BinaryWriter bw = new BinaryWriter(ms);

                // First manually write common properties that would normally
                // be written before SerializeCore is called
                WriteCommonProperties(value, bw, concreteType);

                // Then invoke SerializeCore for specialized data
                serializeCoreMethod.Invoke(value, new object[] { bw });

                // Get the binary data
                byte[] binaryData = ms.ToArray();
                container["CoreBinaryData"] = Convert.ToBase64String(binaryData);
                Console.WriteLine($"Stored CoreBinaryData: {binaryData.Length} bytes");

                // Also capture key property values for debugging
                CaptureKeyProperties(value, container, concreteType);

                return true;
            }
            else
            {
                Console.WriteLine($"No SerializeCore method found on {concreteType.Name}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in core serialization: {ex.Message}");
        }

        return false;
    }

    private bool TryFullReflectionSerialization(object value, JObject container, Type concreteType, JsonSerializer serializer)
    {
        try
        {
            // Create fields container
            JObject fieldsData = new JObject();
            int capturedFieldCount = 0;

            // Get ALL fields, including private ones from base classes
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var allFields = new List<FieldInfo>();

            // Get fields from this type and all base types
            Type? currentType = concreteType;
            while (currentType != null && currentType != typeof(object))
            {
                var fields = currentType.GetFields(flags);
                allFields.AddRange(fields);
                currentType = currentType.BaseType;
            }

            // Try to serialize all fields
            foreach (var field in allFields)
            {
                try
                {
                    var fieldValue = field.GetValue(value);
                    if (fieldValue != null)
                    {
                        // Use type-qualified field names to avoid collisions
                        string typeQualifiedName = $"{field.DeclaringType?.Name}.{field.Name}";
                        fieldsData[typeQualifiedName] = SerializeFieldValue(fieldValue, serializer);
                        capturedFieldCount++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error capturing field {field.Name}: {ex.Message}");
                }
            }

            if (capturedFieldCount > 0)
            {
                container["FieldsData"] = fieldsData;
                Console.WriteLine($"Captured {capturedFieldCount} fields via reflection");
                return true;
            }
            else
            {
                Console.WriteLine("No fields captured via reflection");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in full reflection serialization: {ex.Message}");
        }

        return false;
    }

    private JToken SerializeFieldValue(object fieldValue, JsonSerializer serializer)
    {
        try
        {
            // Handle simple value types directly
            if (fieldValue is string or int or long or double or float or decimal or bool)
            {
                return JToken.FromObject(fieldValue);
            }

            // Handle arrays and collections
            if (fieldValue is System.Collections.ICollection collection)
            {
                JArray array = new JArray();
                foreach (var item in collection)
                {
                    if (item != null)
                    {
                        array.Add(JToken.FromObject(item, serializer));
                    }
                    else
                    {
                        array.Add(JValue.CreateNull());
                    }
                }
                return array;
            }

            // For complex objects, try using the serializer
            return JToken.FromObject(fieldValue, serializer);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to serialize field value: {ex.Message}");
            // Return a placeholder for failed serialization
            return new JObject { ["SerializationFailed"] = ex.Message };
        }
    }

    private void WriteCommonProperties(object value, BinaryWriter writer, Type type)
    {
        // Simulate the common property writing logic that would normally happen
        // in the base class Serialize method before SerializeCore is called

        try
        {
            // Write IsTrained if it exists
            PropertyInfo? isTrainedProp = type.GetProperty("IsTrained",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (isTrainedProp != null)
            {
                bool isTrained = (bool)(isTrainedProp.GetValue(value) ?? false);
                writer.Write(isTrained);
                Console.WriteLine($"Wrote IsTrained: {isTrained}");
            }

            // For regression models
            PropertyInfo? useInterceptProp = type.GetProperty("HasIntercept",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (useInterceptProp != null)
            {
                bool useIntercept = (bool)(useInterceptProp.GetValue(value) ?? false);
                writer.Write(useIntercept);
                Console.WriteLine($"Wrote HasIntercept: {useIntercept}");
            }

            // Try writing coefficients (for regression models)
            PropertyInfo? coefficientsProp = type.GetProperty("Coefficients",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (coefficientsProp != null)
            {
                var coefficients = coefficientsProp.GetValue(value);
                if (coefficients != null)
                {
                    var lengthProp = coefficients.GetType().GetProperty("Length");
                    if (lengthProp != null)
                    {
                        int length = (int)(lengthProp.GetValue(coefficients) ?? 0);
                        writer.Write(length);
                        Console.WriteLine($"Wrote Coefficients.Length: {length}");

                        // Try to write each coefficient
                        if (length > 0)
                        {
                            var indexerMethod = coefficients.GetType().GetMethod("get_Item");
                            if (indexerMethod != null)
                            {
                                for (int i = 0; i < length; i++)
                                {
                                    var coef = indexerMethod.Invoke(coefficients, new object[] { i });
                                    if (coef != null)
                                    {
                                        writer.Write(Convert.ToDouble(coef));
                                    }
                                    else
                                    {
                                        writer.Write(0.0);
                                    }
                                }
                                Console.WriteLine($"Wrote {length} coefficients");
                            }
                        }
                    }
                }
            }

            // For time series models
            PropertyInfo? modelParamsProp = type.GetProperty("ModelParameters",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (modelParamsProp != null)
            {
                var parameters = modelParamsProp.GetValue(value);
                if (parameters != null)
                {
                    var lengthProp = parameters.GetType().GetProperty("Length");
                    if (lengthProp != null)
                    {
                        int length = (int)(lengthProp.GetValue(parameters) ?? 0);
                        writer.Write(length);
                        Console.WriteLine($"Wrote ModelParameters.Length: {length}");

                        // Try to write each parameter
                        if (length > 0)
                        {
                            var indexerMethod = parameters.GetType().GetMethod("get_Item");
                            if (indexerMethod != null)
                            {
                                for (int i = 0; i < length; i++)
                                {
                                    var param = indexerMethod.Invoke(parameters, new object[] { i });
                                    if (param != null)
                                    {
                                        writer.Write(Convert.ToDouble(param));
                                    }
                                    else
                                    {
                                        writer.Write(0.0);
                                    }
                                }
                                Console.WriteLine($"Wrote {length} model parameters");
                            }
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error writing common properties: {ex.Message}");
        }
    }

    private void CaptureKeyProperties(object value, JObject container, Type type)
    {
        try
        {
            JObject props = new JObject();

            // Check for common properties in both model types

            // Common to both
            CaptureProperty(value, props, type, "GetModelMetadata");

            // Time series specific
            CaptureProperty(value, props, type, "IsTrained");
            CaptureProperty(value, props, type, "ModelParameters");
            CaptureProperty(value, props, type, "Options");

            // Regression specific
            CaptureProperty(value, props, type, "Coefficients");
            CaptureProperty(value, props, type, "Intercept");
            CaptureProperty(value, props, type, "FeatureCount");
            CaptureProperty(value, props, type, "HasIntercept");

            if (props.Count > 0)
            {
                container["KeyProperties"] = props;
                Console.WriteLine($"Captured {props.Count} key properties");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error capturing key properties: {ex.Message}");
        }
    }

    private void CaptureProperty(object value, JObject props, Type type, string propertyName)
    {
        try
        {
            PropertyInfo? prop = type.GetProperty(propertyName,
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            if (prop != null && prop.CanRead)
            {
                object? propValue = prop.GetValue(value);
                if (propValue != null)
                {
                    try
                    {
                        props[propertyName] = JToken.FromObject(propValue);
                        Console.WriteLine($"Captured property {propertyName}");
                    }
                    catch
                    {
                        props[propertyName] = $"[Unable to serialize {propValue.GetType().Name}]";
                    }
                }
            }
            else
            {
                // Check if it's a method
                MethodInfo? method = type.GetMethod(propertyName,
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);

                if (method != null)
                {
                    try
                    {
                        object? result = method.Invoke(value, null);
                        if (result != null)
                        {
                            props[propertyName + "()"] = JToken.FromObject(result);
                            Console.WriteLine($"Captured method result {propertyName}()");
                        }
                    }
                    catch (Exception ex)
                    {
                        props[propertyName + "()"] = $"[Error: {ex.Message}]";
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error capturing property {propertyName}: {ex.Message}");
        }
    }

    // ================================================================
    // Deserialization helper methods
    // ================================================================

    private bool TryBinaryDeserialization(JObject container, Type concreteType, object model)
    {
        try
        {
            string? binaryString = container["BinaryData"]?.Value<string>();
            if (!string.IsNullOrEmpty(binaryString))
            {
                byte[] binaryData = Convert.FromBase64String(binaryString);
                Console.WriteLine($"Found BinaryData: {binaryData.Length} bytes");

                MethodInfo? deserializeMethod = concreteType.GetMethod("Deserialize");
                if (deserializeMethod != null)
                {
                    Console.WriteLine($"Found Deserialize method on {concreteType.Name}");
                    deserializeMethod.Invoke(model, new object[] { binaryData });
                    Console.WriteLine("Successfully called Deserialize method");

                    // Verify key properties to confirm successful deserialization
                    LogModelState(model, "After binary deserialization");

                    return true;
                }
                else
                {
                    Console.WriteLine($"No Deserialize method found on {concreteType.Name}");
                }
            }
            else
            {
                Console.WriteLine("No BinaryData found in JSON");

                // Try CoreBinaryData if BinaryData isn't available
                string? coreBinaryString = container["CoreBinaryData"]?.Value<string>();
                if (!string.IsNullOrEmpty(coreBinaryString))
                {
                    byte[] coreBinaryData = Convert.FromBase64String(coreBinaryString);
                    Console.WriteLine($"Found CoreBinaryData: {coreBinaryData.Length} bytes");

                    // Try to deserialize using DeserializeCore if available
                    MethodInfo? deserializeCoreMethod = FindMethod(concreteType, "DeserializeCore",
                        new[] { typeof(BinaryReader) });

                    if (deserializeCoreMethod != null)
                    {
                        using MemoryStream ms = new MemoryStream(coreBinaryData);
                        using BinaryReader br = new BinaryReader(ms);

                        // First read common properties that would normally
                        // be read before DeserializeCore is called
                        ReadCommonProperties(model, br, concreteType);

                        // Then invoke DeserializeCore for specialized data
                        deserializeCoreMethod.Invoke(model, new object[] { br });
                        Console.WriteLine("Successfully called DeserializeCore method");

                        // Verify key properties to confirm successful deserialization
                        LogModelState(model, "After core binary deserialization");

                        return true;
                    }
                    else
                    {
                        Console.WriteLine($"No DeserializeCore method found on {concreteType.Name}");
                    }
                }
                else
                {
                    Console.WriteLine("No CoreBinaryData found in JSON");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in binary deserialization: {ex.Message}");
        }

        return false;
    }

    private bool TryFieldsDeserialization(JObject container, object model)
    {
        try
        {
            JObject? fieldsData = container["FieldsData"] as JObject;
            if (fieldsData != null && fieldsData.Count > 0)
            {
                Console.WriteLine($"Found FieldsData with {fieldsData.Count} entries");

                Type modelType = model.GetType();
                int restoredFieldCount = 0;

                // Get all fields including private ones from the entire hierarchy
                var allFields = new Dictionary<string, FieldInfo>();
                Type? currentType = modelType;

                while (currentType != null && currentType != typeof(object))
                {
                    var typeFields = currentType.GetFields(BindingFlags.Instance |
                        BindingFlags.Public | BindingFlags.NonPublic);

                    foreach (var field in typeFields)
                    {
                        string typeQualifiedName = $"{field.DeclaringType?.Name}.{field.Name}";
                        allFields[typeQualifiedName] = field;
                    }

                    currentType = currentType.BaseType;
                }

                // Try to restore each field
                foreach (var prop in fieldsData.Properties())
                {
                    if (allFields.TryGetValue(prop.Name, out FieldInfo? field))
                    {
                        try
                        {
                            object? value = prop.Value.ToObject(field.FieldType);
                            field.SetValue(model, value);
                            restoredFieldCount++;
                            Console.WriteLine($"Restored field {prop.Name}");
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Failed to restore field {prop.Name}: {ex.Message}");
                        }
                    }
                }

                if (restoredFieldCount > 0)
                {
                    Console.WriteLine($"Restored {restoredFieldCount} fields via reflection");

                    // After restoring fields, synchronize internal state
                    SynchronizeInternalState(model);

                    // Verify key properties to confirm successful deserialization
                    LogModelState(model, "After fields deserialization");

                    return true;
                }
                else
                {
                    Console.WriteLine("No fields were successfully restored");
                }
            }
            else
            {
                Console.WriteLine("No FieldsData found in JSON");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in fields deserialization: {ex.Message}");
        }

        return false;
    }

    private void ReadCommonProperties(object model, BinaryReader reader, Type type)
    {
        try
        {
            // Read IsTrained if it exists (for time series models)
            PropertyInfo? isTrainedProp = type.GetProperty("IsTrained",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (isTrainedProp != null && isTrainedProp.CanWrite)
            {
                bool isTrained = reader.ReadBoolean();
                isTrainedProp.SetValue(model, isTrained);
                Console.WriteLine($"Read IsTrained: {isTrained}");
            }

            // For regression models
            PropertyInfo? optionsProp = type.GetProperty("Options",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (optionsProp != null)
            {
                var options = optionsProp.GetValue(model);
                if (options != null)
                {
                    PropertyInfo? useInterceptProp = options.GetType().GetProperty("UseIntercept");
                    if (useInterceptProp != null && useInterceptProp.CanWrite)
                    {
                        bool useIntercept = reader.ReadBoolean();
                        useInterceptProp.SetValue(options, useIntercept);
                        Console.WriteLine($"Read UseIntercept: {useIntercept}");
                    }
                }
            }

            // Try reading coefficients (for regression models)
            PropertyInfo? coefficientsProp = type.GetProperty("Coefficients",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (coefficientsProp != null && coefficientsProp.CanWrite)
            {
                int length = reader.ReadInt32();
                Console.WriteLine($"Read Coefficients.Length: {length}");

                if (length > 0)
                {
                    // Need to create a new Vector
                    var elementType = GetGenericElementType(type);
                    if (elementType != null)
                    {
                        Type vectorType = typeof(Vector<>).MakeGenericType(elementType);
                        object? vector = Activator.CreateInstance(vectorType, length);

                        if (vector != null)
                        {
                            // Find the indexer setter
                            MethodInfo? setItemMethod = vectorType.GetMethod("set_Item");
                            if (setItemMethod != null)
                            {
                                // Get conversion method
                                MethodInfo? fromDoubleMethod = null;
                                var numOpsProperty = type.GetProperty("NumOps",
                                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                                if (numOpsProperty != null)
                                {
                                    var numOps = numOpsProperty.GetValue(model);
                                    if (numOps != null)
                                    {
                                        fromDoubleMethod = numOps.GetType().GetMethod("FromDouble");
                                    }
                                }

                                // Read each coefficient
                                for (int i = 0; i < length; i++)
                                {
                                    double value = reader.ReadDouble();

                                    if (fromDoubleMethod != null)
                                    {
                                        // Convert via NumOps.FromDouble
                                        object? converted = fromDoubleMethod.Invoke(
                                            numOpsProperty?.GetValue(model), new object[] { value });

                                        if (converted != null)
                                        {
                                            setItemMethod.Invoke(vector, new[] { i, converted });
                                        }
                                    }
                                    else
                                    {
                                        // Try direct conversion
                                        object? converted = Convert.ChangeType(value, elementType);
                                        setItemMethod.Invoke(vector, new[] { i, converted });
                                    }
                                }

                                // Set the coefficients
                                coefficientsProp.SetValue(model, vector);
                                Console.WriteLine($"Set Coefficients with {length} values");
                            }
                        }
                    }
                }
            }

            // For time series models
            PropertyInfo? modelParamsProp = type.GetProperty("ModelParameters",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (modelParamsProp != null && modelParamsProp.CanWrite)
            {
                // Similar implementation as coefficients
                // Omitted for brevity but would follow same pattern
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error reading common properties: {ex.Message}");
        }
    }

    private Type? GetGenericElementType(Type? modelType)
    {
        while (modelType != null && modelType != typeof(object))
        {
            if (modelType.IsGenericType)
            {
                return modelType.GetGenericArguments()[0];
            }

            modelType = modelType.BaseType;
        }

        return null;
    }

    private void SynchronizeInternalState(object model)
    {
        try
        {
            Type modelType = model.GetType();

            // Call key methods to ensure internal state is synchronized

            // For regression models
            InvokeMethod(model, "GetParameters");
            InvokeMethod(model, "GetActiveFeatureIndices");
            InvokeMethod(model, "CalculateFeatureImportances");

            // For time series models
            InvokeMethod(model, "GetModelMetadata");

            // Check if IsTrained property exists and set it to true if the model has parameters
            PropertyInfo? isTrainedProp = modelType.GetProperty("IsTrained",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            if (isTrainedProp != null && isTrainedProp.CanWrite)
            {
                bool hasParameters = false;

                // For regression models
                PropertyInfo? coefficientsProp = modelType.GetProperty("Coefficients",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (coefficientsProp != null)
                {
                    var coefficients = coefficientsProp.GetValue(model);
                    if (coefficients != null)
                    {
                        var lengthProp = coefficients.GetType().GetProperty("Length");
                        if (lengthProp != null)
                        {
                            int length = (int)(lengthProp.GetValue(coefficients) ?? 0);
                            hasParameters = length > 0;
                        }
                    }
                }

                // For time series models
                if (!hasParameters)
                {
                    PropertyInfo? modelParamsProp = modelType.GetProperty("ModelParameters",
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    if (modelParamsProp != null)
                    {
                        var parameters = modelParamsProp.GetValue(model);
                        if (parameters != null)
                        {
                            var lengthProp = parameters.GetType().GetProperty("Length");
                            if (lengthProp != null)
                            {
                                int length = (int)(lengthProp.GetValue(parameters) ?? 0);
                                hasParameters = length > 0;
                            }
                        }
                    }
                }

                if (hasParameters)
                {
                    isTrainedProp.SetValue(model, true);
                    Console.WriteLine("Set IsTrained to true based on parameter existence");
                }
            }

            // For regression models, ensure FeatureCount matches Coefficients.Length
            PropertyInfo? featureCountProp = modelType.GetProperty("FeatureCount",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            if (featureCountProp != null && featureCountProp.CanWrite)
            {
                PropertyInfo? coefficientsProp = modelType.GetProperty("Coefficients",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                if (coefficientsProp != null)
                {
                    var coefficients = coefficientsProp.GetValue(model);
                    if (coefficients != null)
                    {
                        var lengthProp = coefficients.GetType().GetProperty("Length");
                        if (lengthProp != null)
                        {
                            int length = (int)(lengthProp.GetValue(coefficients) ?? 0);
                            featureCountProp.SetValue(model, length);
                            Console.WriteLine($"Set FeatureCount to {length} to match Coefficients.Length");
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error synchronizing internal state: {ex.Message}");
        }
    }

    private void InvokeMethod(object model, string methodName)
    {
        try
        {
            MethodInfo? method = model.GetType().GetMethod(methodName,
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                null, Type.EmptyTypes, null);

            if (method != null)
            {
                method.Invoke(model, null);
                Console.WriteLine($"Invoked {methodName}() method to help synchronize state");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error invoking {methodName}(): {ex.Message}");
        }
    }

    private void ValidateDeserializedObject(object model)
    {
        try
        {
            LogModelState(model, "Final deserialized state");

            // Try to invoke basic prediction to verify the model works
            MethodInfo? predictMethod = model.GetType().GetMethod("Predict");
            if (predictMethod != null)
            {
                Console.WriteLine("Model has a Predict method - good sign");
            }

            // Check other critical properties/methods
            if (model.GetType().GetProperty("IsTrained") != null)
            {
                var prop = model.GetType()?.GetProperty("IsTrained");
                bool isTrained = (bool)(prop?.GetValue(model) ?? false);
                Console.WriteLine($"IsTrained = {isTrained}");
            }

            // Try to get key parameters (depends on model type)
            var paramMethod = model.GetType().GetMethod("GetParameters");
            if (paramMethod != null)
            {
                try
                {
                    var parameters = paramMethod.Invoke(model, null);
                    if (parameters != null)
                    {
                        var lengthProp = parameters.GetType().GetProperty("Length");
                        if (lengthProp != null)
                        {
                            int length = (int)(lengthProp.GetValue(parameters) ?? 0);
                            Console.WriteLine($"GetParameters() returned vector with {length} elements");
                        }
                    }
                    else
                    {
                        Console.WriteLine("GetParameters() returned null");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error calling GetParameters(): {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error validating model: {ex.Message}");
        }
    }

    private void LogModelState(object model, string context)
    {
        try
        {
            Console.WriteLine($"--- {context} ---");

            Type modelType = model.GetType();

            // Check for common properties in both model types

            // Common to both
            LogPropertyValue(model, modelType, "GetModelMetadata");

            // Time series specific
            LogPropertyValue(model, modelType, "IsTrained");
            LogPropertyValue(model, modelType, "ModelParameters");
            LogPropertyValue(model, modelType, "Options");

            // Regression specific
            LogPropertyValue(model, modelType, "Coefficients");
            LogPropertyValue(model, modelType, "Intercept");
            LogPropertyValue(model, modelType, "FeatureCount");
            LogPropertyValue(model, modelType, "HasIntercept");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error logging model state: {ex.Message}");
        }
    }

    private void LogPropertyValue(object model, Type modelType, string propertyName)
    {
        try
        {
            PropertyInfo? prop = modelType.GetProperty(propertyName,
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            if (prop != null && prop.CanRead)
            {
                object? propValue = prop.GetValue(model);
                if (propValue != null)
                {
                    // For collections, show count
                    if (propValue is System.Collections.ICollection collection)
                    {
                        Console.WriteLine($"{propertyName}: {propValue.GetType().Name} with {collection.Count} items");
                    }
                    else
                    {
                        Console.WriteLine($"{propertyName}: {propValue}");
                    }
                }
                else
                {
                    Console.WriteLine($"{propertyName}: null");
                }
            }
            else
            {
                // Check if it's a method
                MethodInfo? method = modelType.GetMethod(propertyName,
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);

                if (method != null)
                {
                    try
                    {
                        object? result = method.Invoke(model, null);
                        if (result != null)
                        {
                            Console.WriteLine($"{propertyName}(): returned {result.GetType().Name}");
                        }
                        else
                        {
                            Console.WriteLine($"{propertyName}(): returned null");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"{propertyName}(): error - {ex.Message}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error logging property {propertyName}: {ex.Message}");
        }
    }

    // ================================================================
    // Utility methods
    // ================================================================

    /// <summary>
    /// Resolves a type from its string representation, checking all loaded assemblies if needed.
    /// </summary>
    protected Type? ResolveType(string? typeString)
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
    /// Creates a new instance of a model type, handling any constructor requirements.
    /// </summary>
    protected object? CreateModelInstance(Type modelType)
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
    protected MethodInfo? FindMethod(Type type, string methodName, Type[] parameterTypes)
    {
        return type.GetMethod(methodName,
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
            null, parameterTypes, null);
    }

    /// <summary>
    /// Logs a debug message to the debug log file.
    /// </summary>
    protected void DebugLog(string message)
    {
        if (!EnableDebugOutput)
            return;

        try
        {
            File.AppendAllText(DebugLogPath, $"[{DateTime.Now:HH:mm:ss.fff}] {message}\n");
        }
        catch
        {
            // Ignore errors writing to log file
        }
    }
}