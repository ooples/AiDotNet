using AiDotNet.Interfaces;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides static methods for loading self-describing AIMF model files with automatic type detection.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When you save a model using the new AIMF envelope format, the file contains
/// metadata about what type of model it is. ModelLoader reads this metadata and automatically creates
/// the correct model instance, so you don't need to know the exact model type in advance.
///
/// Example usage:
/// <code>
/// // Save a model (automatically adds AIMF header)
/// myNetwork.SaveModel("model.aimf");
///
/// // Load it back - ModelLoader figures out the type automatically
/// var model = ModelLoader.Load&lt;double&gt;("model.aimf");
///
/// // Inspect a model file without loading it
/// var info = ModelLoader.Inspect("model.aimf");
/// Console.WriteLine($"Model type: {info.TypeName}");
/// Console.WriteLine($"Input shape: [{string.Join(", ", info.InputShape)}]");
/// </code>
///
/// All model files must use the AIMF envelope format. Files saved with SaveModel()
/// automatically include the AIMF header.
/// </remarks>
public static class ModelLoader
{
    /// <summary>
    /// Loads a self-describing AIMF model file, automatically detecting and instantiating the correct model type.
    /// </summary>
    /// <typeparam name="T">
    /// The numeric type used by the model (e.g., double, float).
    /// This must match the type the model was originally trained with.
    /// </typeparam>
    /// <param name="filePath">The path to the AIMF model file.</param>
    /// <returns>The deserialized model instance.</returns>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the file does not have an AIMF header, or the model type cannot be resolved.
    /// </exception>
    public static IModelSerializer Load<T>(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        byte[] data = File.ReadAllBytes(filePath);
        return LoadFromBytes<T>(data);
    }

    /// <summary>
    /// Loads a self-describing AIMF model from a byte array.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="data">The byte array containing the AIMF-enveloped model.</param>
    /// <returns>The deserialized model instance.</returns>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the data does not have an AIMF header, or the model type cannot be resolved.
    /// </exception>
    public static IModelSerializer LoadFromBytes<T>(byte[] data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (!ModelFileHeader.HasHeader(data))
        {
            throw new InvalidOperationException(
                "The data does not contain an AIMF envelope header. " +
                "This is not a valid AIMF model file. " +
                "Models must be saved with the AIMF envelope format.");
        }

        // Read header
        var info = ModelFileHeader.ReadHeader(data);

        // Resolve the model type
        var openGenericType = ModelTypeRegistry.Resolve(info.TypeName, info.AssemblyQualifiedName);
        if (openGenericType is null)
        {
            throw new InvalidOperationException(
                $"Cannot resolve model type '{info.TypeName}'. " +
                $"Assembly-qualified name: '{info.AssemblyQualifiedName}'. " +
                "The model type may not be available in the current assembly. " +
                "Register external model types using ModelTypeRegistry.Register() or ModelTypeRegistry.RegisterAssembly().");
        }

        // Create an instance
        var model = ModelTypeRegistry.CreateInstance<T>(openGenericType);

        // Extract payload and deserialize
        byte[] payload = ModelFileHeader.ExtractPayload(data, info);
        model.Deserialize(payload);

        // Optional shape validation if the model implements IModelShape
        if (model is IModelShape shapeModel)
        {
            ValidateShapes(shapeModel, info);
        }

        return model;
    }

    /// <summary>
    /// Reads only the header of an AIMF model file without loading the full model.
    /// This is useful for inspecting model metadata (type, shapes) without the cost of full deserialization.
    /// </summary>
    /// <param name="filePath">The path to the AIMF model file.</param>
    /// <returns>A ModelFileInfo containing the header metadata.</returns>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the file does not have an AIMF header.</exception>
    public static ModelFileInfo Inspect(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        byte[] data = File.ReadAllBytes(filePath);
        return ModelFileHeader.ReadHeader(data);
    }

    /// <summary>
    /// Checks whether a file is a self-describing AIMF model file by reading only the first 4 bytes.
    /// </summary>
    /// <param name="filePath">The path to the file to check.</param>
    /// <returns>True if the file starts with the AIMF magic bytes.</returns>
    public static bool IsSelfDescribing(string filePath)
    {
        return ModelFileHeader.HasHeader(filePath);
    }

    /// <summary>
    /// Validates that the deserialized model's shapes match those stored in the header.
    /// Issues a diagnostic warning if they don't match (does not throw).
    /// </summary>
    private static void ValidateShapes(IModelShape model, ModelFileInfo headerInfo)
    {
        // Only validate if header had shape info (non-empty arrays)
        if (headerInfo.InputShape.Length == 0 && headerInfo.OutputShape.Length == 0)
        {
            return;
        }

        var actualInput = model.GetInputShape();
        var actualOutput = model.GetOutputShape();

        if (headerInfo.InputShape.Length > 0 && !ShapesMatch(actualInput, headerInfo.InputShape))
        {
            System.Diagnostics.Debug.WriteLine(
                $"[ModelLoader] Warning: Input shape mismatch. " +
                $"Header: [{string.Join(", ", headerInfo.InputShape)}], " +
                $"Model: [{string.Join(", ", actualInput)}]");
        }

        if (headerInfo.OutputShape.Length > 0 && !ShapesMatch(actualOutput, headerInfo.OutputShape))
        {
            System.Diagnostics.Debug.WriteLine(
                $"[ModelLoader] Warning: Output shape mismatch. " +
                $"Header: [{string.Join(", ", headerInfo.OutputShape)}], " +
                $"Model: [{string.Join(", ", actualOutput)}]");
        }
    }

    private static bool ShapesMatch(int[] a, int[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            // -1 means dynamic dimension, always matches
            if (a[i] == -1 || b[i] == -1)
            {
                continue;
            }

            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}
