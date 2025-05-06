global using Newtonsoft.Json.Linq;

namespace AiDotNet.Serialization;

/// <summary>
/// Custom JSON converter for the Matrix&lt;T&gt; class to enable proper serialization and deserialization.
/// </summary>
/// <typeparam name="T">The numeric type of the matrix elements.</typeparam>
/// <remarks>
/// <para>
/// This converter handles the serialization and deserialization of Matrix&lt;T&gt; objects using Newtonsoft.Json.
/// It leverages the existing SerializationHelper to maintain consistency with the binary serialization format.
/// </para>
/// <para><b>For Beginners:</b> This converter tells the JSON serializer how to save and load our special Matrix type.
/// 
/// When working with complex types like our Matrix class:
/// - The default JSON serializer doesn't know how to handle them properly
/// - We need to provide custom instructions for saving and loading matrices
/// - This converter provides those instructions by using our existing serialization helpers
/// 
/// This ensures consistency between our binary and JSON serialization formats.
/// </para>
/// </remarks>
public class MatrixJsonConverter<T> : JsonConverter
{
    /// <summary>
    /// Determines whether this converter can convert the specified object type.
    /// </summary>
    /// <param name="objectType">The type of the object to check.</param>
    /// <returns>True if this converter can convert the specified type; otherwise, false.</returns>
    public override bool CanConvert(Type objectType)
    {
        return objectType == typeof(Matrix<T>);
    }

    /// <summary>
    /// Writes the JSON representation of the Matrix&lt;T&gt; object.
    /// </summary>
    /// <param name="writer">The JsonWriter to write to.</param>
    /// <param name="value">The Matrix&lt;T&gt; to convert to JSON.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
    {
        if (value is not Matrix<T> matrix)
        {
            writer.WriteNull();
            return;
        }

        // Convert the matrix to a byte array using our existing serialization helper
        byte[] serializedData;
        using (var ms = new MemoryStream())
        using (var bw = new BinaryWriter(ms))
        {
            SerializationHelper<T>.SerializeMatrix(bw, matrix);
            serializedData = ms.ToArray();
        }

        // Create a JSON object with matrix dimensions and Base64-encoded data
        JObject obj = new JObject();
        obj["Rows"] = matrix.Rows;
        obj["Columns"] = matrix.Columns;
        obj["Data"] = Convert.ToBase64String(serializedData);

        obj.WriteTo(writer);
    }

    /// <summary>
    /// Reads the JSON representation of the Matrix&lt;T&gt; object.
    /// </summary>
    /// <param name="reader">The JsonReader to read from.</param>
    /// <param name="objectType">The type of the object to convert.</param>
    /// <param name="existingValue">The existing value of the object being read.</param>
    /// <param name="serializer">The JsonSerializer to use.</param>
    /// <returns>The Matrix&lt;T&gt; object deserialized from JSON.</returns>
    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null)
            return null;

        JObject obj = JObject.Load(reader);

        int rows = obj["Rows"]?.Value<int>() ?? 0;
        int columns = obj["Columns"]?.Value<int>() ?? 0;

        if (rows == 0 || columns == 0)
            return new Matrix<T>(0, 0);

        string? base64Data = obj["Data"]?.Value<string>();
        if (string.IsNullOrEmpty(base64Data))
            return new Matrix<T>(rows, columns);

        // Decode and deserialize using our existing helper
        byte[] binaryData = Convert.FromBase64String(base64Data);
        Matrix<T> matrix;

        using (var ms = new MemoryStream(binaryData))
        using (var br = new BinaryReader(ms))
        {
            // Skip the dimensions since we already read them from JSON
            // and SerializationHelper writes them at the beginning
            br.ReadInt32(); // Skip stored rows
            br.ReadInt32(); // Skip stored columns

            // Create the matrix with the dimensions we got from JSON
            matrix = new Matrix<T>(rows, columns);

            // Read all the values
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    matrix[i, j] = SerializationHelper<T>.ReadValue(br);
                }
            }
        }

        return matrix;
    }
}