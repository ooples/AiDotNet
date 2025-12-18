using System;
using AiDotNet.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serialization
{
    /// <summary>
    /// JSON converter for Matrix&lt;T&gt; types.
    /// Handles serialization and deserialization of matrix objects to/from JSON.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This class knows how to convert a Matrix (a grid of numbers) into
    /// JSON text format and back. It saves the number of rows, columns, and all the data values,
    /// so the matrix can be perfectly reconstructed later.</para>
    /// </remarks>
    public class MatrixJsonConverter : JsonConverter
    {
        /// <summary>
        /// Determines whether this converter can handle the specified type.
        /// </summary>
        /// <param name="objectType">The type to check.</param>
        /// <returns>True if the type is Matrix&lt;T&gt; or a subclass thereof, false otherwise.</returns>
        /// <remarks>
        /// This method walks the inheritance chain to support subclasses of Matrix&lt;T&gt;.
        /// </remarks>
        public override bool CanConvert(Type objectType)
        {
            // Walk the inheritance chain to support subclasses
            Type? currentType = objectType;
            while (currentType != null)
            {
                if (currentType.IsGenericType &&
                    currentType.GetGenericTypeDefinition() == typeof(Matrix<>))
                {
                    return true;
                }
                currentType = currentType.BaseType;
            }
            return false;
        }

        /// <summary>
        /// Writes a Matrix&lt;T&gt; object to JSON.
        /// </summary>
        /// <param name="writer">The JSON writer.</param>
        /// <param name="value">The matrix to serialize.</param>
        /// <param name="serializer">The JSON serializer.</param>
        /// <remarks>
        /// <para><b>For Beginners:</b> This method converts a Matrix into JSON format by saving:
        /// 1. The number of rows
        /// 2. The number of columns
        /// 3. All the data in the matrix
        /// This allows the matrix to be saved to a file.</para>
        /// </remarks>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value == null)
            {
                writer.WriteNull();
                return;
            }

            var matrixType = value.GetType();
            var rowsProperty = matrixType.GetProperty("Rows");
            var columnsProperty = matrixType.GetProperty("Columns");
            var indexer = matrixType.GetProperty("Item", new[] { typeof(int), typeof(int) });

            if (rowsProperty == null || columnsProperty == null || indexer == null)
            {
                throw new JsonSerializationException($"Cannot serialize matrix type {matrixType.Name}: missing required properties.");
            }

            object? rowsObj = rowsProperty.GetValue(value);
            object? columnsObj = columnsProperty.GetValue(value);
            if (rowsObj == null || columnsObj == null)
            {
                throw new JsonSerializationException($"Cannot serialize matrix: Rows or Columns property returned null.");
            }
            var rows = (int)rowsObj;
            var columns = (int)columnsObj;

            writer.WriteStartObject();
            writer.WritePropertyName("rows");
            writer.WriteValue(rows);
            writer.WritePropertyName("columns");
            writer.WriteValue(columns);
            writer.WritePropertyName("data");
            writer.WriteStartArray();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    var cellValue = indexer.GetValue(value, new object[] { i, j });
                    serializer.Serialize(writer, cellValue);
                }
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        /// <summary>
        /// Reads a Matrix&lt;T&gt; object from JSON.
        /// </summary>
        /// <param name="reader">The JSON reader.</param>
        /// <param name="objectType">The type of object to create.</param>
        /// <param name="existingValue">The existing value (not used).</param>
        /// <param name="serializer">The JSON serializer.</param>
        /// <returns>A reconstructed Matrix&lt;T&gt; object.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> This method reads JSON data and reconstructs a Matrix object.
        /// It reads the rows, columns, and data that were saved, then creates a new matrix with
        /// those exact values.</para>
        /// </remarks>
        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null)
            {
                return null;
            }

            var jObject = JObject.Load(reader);

            // Validate required tokens exist
            var rowsToken = jObject["rows"];
            var columnsToken = jObject["columns"];
            var dataArray = jObject["data"] as JArray;

            if (rowsToken == null || columnsToken == null || dataArray == null)
            {
                throw new JsonSerializationException("Matrix JSON must contain 'rows', 'columns', and 'data' (array).");
            }

            var rows = rowsToken.Value<int>();
            var columns = columnsToken.Value<int>();

            // Validate dimensions are non-negative
            if (rows < 0 || columns < 0)
            {
                throw new JsonSerializationException("Matrix 'rows' and 'columns' must be non-negative.");
            }

            // Validate data length matches dimensions
            int expectedLength = rows * columns;
            if (dataArray.Count != expectedLength)
            {
                throw new JsonSerializationException(
                    $"Matrix data length {dataArray.Count} does not match rows*columns {expectedLength}.");
            }

            // Get the element type (T) from Matrix<T>
            var elementType = objectType.GetGenericArguments()[0];

            // Create matrix constructor: Matrix<T>(int rows, int columns)
            var matrixConstructor = objectType.GetConstructor(new[] { typeof(int), typeof(int) });
            if (matrixConstructor == null)
            {
                throw new JsonSerializationException($"Cannot find constructor for {objectType.Name}(int, int)");
            }

            var matrix = matrixConstructor.Invoke(new object[] { rows, columns });

            // Get the indexer property for setting values
            var indexer = objectType.GetProperty("Item", new[] { typeof(int), typeof(int) });
            if (indexer == null)
            {
                throw new JsonSerializationException($"Cannot find indexer for {objectType.Name}");
            }

            // Populate the matrix using the provided serializer
            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    var value = dataArray[index++].ToObject(elementType, serializer);
                    indexer.SetValue(matrix, value, new object[] { i, j });
                }
            }

            return matrix;
        }
    }
}
