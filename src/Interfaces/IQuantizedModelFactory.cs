using System.IO;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for factories that can deserialize quantized models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IQuantizedModelFactory<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Deserializes a quantized model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>The deserialized quantized model.</returns>
    TModel DeserializeQuantizedModel(BinaryReader reader);
}