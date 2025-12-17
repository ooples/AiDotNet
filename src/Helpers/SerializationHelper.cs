namespace AiDotNet.Helpers;

/// <summary>
/// Provides methods for serializing and deserializing AI model components to and from binary formats.
/// </summary>
/// <typeparam name="T">The numeric type used in the AI models (e.g., double, float, int).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Serialization is the process of converting complex data structures (like AI models) 
/// into a format that can be easily stored or transmitted. Think of it like saving your game progress 
/// so you can continue later. This helper makes it possible to save trained AI models to disk and 
/// load them back when needed.
/// </remarks>
public static class SerializationHelper<T>
{
    /// <summary>
    /// Provides operations specific to the numeric type being used.
    /// </summary>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Serializes a decision tree node and its children to a binary format.
    /// </summary>
    /// <param name="node">The decision tree node to serialize.</param>
    /// <param name="writer">The binary writer to write the serialized data to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method saves a decision tree (a flowchart-like model that makes decisions) 
    /// to a file. Decision trees are used in AI for classification and regression tasks.
    /// </remarks>
    public static void SerializeNode(DecisionTreeNode<T>? node, BinaryWriter writer)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.IsLeaf);

        if (node.IsLeaf)
        {
            writer.Write(Convert.ToDouble(node.Prediction));
        }
        else
        {
            writer.Write(node.FeatureIndex);
            writer.Write(Convert.ToDouble(node.SplitValue));
            SerializeNode(node.Left, writer);
            SerializeNode(node.Right, writer);
        }
    }

    /// <summary>
    /// Deserializes a decision tree node and its children from a binary format.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized data from.</param>
    /// <returns>The deserialized decision tree node.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads a previously saved decision tree from a file,
    /// reconstructing its structure so you can use it to make predictions without retraining.
    /// </remarks>
    public static DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        if (!reader.ReadBoolean())
        {
            return null;
        }

        var node = new DecisionTreeNode<T>
        {
            IsLeaf = reader.ReadBoolean()
        };

        if (node.IsLeaf)
        {
            node.Prediction = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
        }
        else
        {
            node.FeatureIndex = reader.ReadInt32();
            node.SplitValue = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
            node.Left = DeserializeNode(reader);
            node.Right = DeserializeNode(reader);
        }

        return node;
    }

    /// <summary>
    /// Writes a value of type T to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="value">The value to write.</param>
    /// <exception cref="NotSupportedException">Thrown when the type T is not supported for serialization.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method handles writing different types of numbers (like integers or decimals)
    /// to a file in a way that preserves their exact values.
    /// </remarks>
    public static void WriteValue(BinaryWriter writer, T value)
    {
        if (typeof(T) == typeof(double))
        {
            writer.Write(Convert.ToDouble(value));
        }
        else if (typeof(T) == typeof(float))
        {
            writer.Write(Convert.ToSingle(value));
        }
        else if (typeof(T) == typeof(decimal))
        {
            writer.Write(Convert.ToDecimal(value));
        }
        else if (typeof(T) == typeof(long))
        {
            writer.Write(Convert.ToInt64(value));
        }
        else if (typeof(T) == typeof(ulong))
        {
            writer.Write(Convert.ToUInt64(value));
        }
        else if (typeof(T) == typeof(int))
        {
            writer.Write(Convert.ToInt32(value));
        }
        else if (typeof(T) == typeof(uint))
        {
            writer.Write(Convert.ToUInt32(value));
        }
        else if (typeof(T) == typeof(short))
        {
            writer.Write(Convert.ToInt16(value));
        }
        else if (typeof(T) == typeof(ushort))
        {
            writer.Write(Convert.ToUInt16(value));
        }
        else if (typeof(T) == typeof(byte))
        {
            writer.Write(Convert.ToByte(value));
        }
        else if (typeof(T) == typeof(sbyte))
        {
            writer.Write(Convert.ToSByte(value));
        }
        else if (typeof(T) == typeof(bool))
        {
            writer.Write(Convert.ToBoolean(value));
        }
        else if (typeof(T) == typeof(char))
        {
            writer.Write(Convert.ToChar(value));
        }
        else
        {
            throw new NotSupportedException($"Serialization of type {typeof(T)} is not supported.");
        }
    }

    /// <summary>
    /// Reads a value of type T from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>The value read from the binary reader.</returns>
    /// <exception cref="NotSupportedException">Thrown when the type T is not supported for deserialization.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method reads different types of numbers from a file and converts them
    /// back to their original form so they can be used in calculations.
    /// </remarks>
    public static T ReadValue(BinaryReader reader)
    {
        if (typeof(T) == typeof(double))
        {
            return (T)(object)reader.ReadDouble();
        }
        else if (typeof(T) == typeof(float))
        {
            return (T)(object)reader.ReadSingle();
        }
        else if (typeof(T) == typeof(decimal))
        {
            return (T)(object)reader.ReadDecimal();
        }
        else if (typeof(T) == typeof(long))
        {
            return (T)(object)reader.ReadInt64();
        }
        else if (typeof(T) == typeof(ulong))
        {
            return (T)(object)reader.ReadUInt64();
        }
        else if (typeof(T) == typeof(int))
        {
            return (T)(object)reader.ReadInt32();
        }
        else if (typeof(T) == typeof(uint))
        {
            return (T)(object)reader.ReadUInt32();
        }
        else if (typeof(T) == typeof(short))
        {
            return (T)(object)reader.ReadInt16();
        }
        else if (typeof(T) == typeof(ushort))
        {
            return (T)(object)reader.ReadUInt16();
        }
        else if (typeof(T) == typeof(byte))
        {
            return (T)(object)reader.ReadByte();
        }
        else if (typeof(T) == typeof(sbyte))
        {
            return (T)(object)reader.ReadSByte();
        }
        else if (typeof(T) == typeof(bool))
        {
            return (T)(object)reader.ReadBoolean();
        }
        else if (typeof(T) == typeof(char))
        {
            return (T)(object)reader.ReadChar();
        }
        else
        {
            throw new NotSupportedException($"Deserialization of type {typeof(T)} is not supported.");
        }
    }

    /// <summary>
    /// Serializes a matrix to a binary format.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized data to.</param>
    /// <param name="matrix">The matrix to serialize.</param>
    /// <remarks>
    /// <b>For Beginners:</b> A matrix is a rectangular grid of numbers used in many AI algorithms.
    /// This method saves that grid to a file so you can use it later.
    /// </remarks>
    public static void SerializeMatrix(BinaryWriter writer, Matrix<T> matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                writer.Write(Convert.ToDouble(matrix[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes a matrix from a binary format with expected dimensions.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized data from.</param>
    /// <param name="rows">The expected number of rows in the matrix.</param>
    /// <param name="columns">The expected number of columns in the matrix.</param>
    /// <returns>The deserialized matrix.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the stored matrix dimensions do not match the expected dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads a previously saved matrix from a file, but checks that
    /// it has the size you expect. This is important because AI algorithms need matrices of specific sizes.
    /// </remarks>
    public static Matrix<T> DeserializeMatrix(BinaryReader reader, int rows, int columns)
    {
        int storedRows = reader.ReadInt32();
        int storedColumns = reader.ReadInt32();

        if (storedRows != rows || storedColumns != columns)
        {
            throw new InvalidOperationException("Stored matrix dimensions do not match expected dimensions.");
        }

        Matrix<T> matrix = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = _numOps.FromDouble(reader.ReadDouble());
            }
        }

        return matrix;
    }

    /// <summary>
    /// Serializes a matrix to a byte array.
    /// </summary>
    /// <param name="matrix">The matrix to serialize.</param>
    /// <returns>A byte array containing the serialized matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts a matrix (grid of numbers) into a compact format
    /// that can be easily stored in memory or in a database.
    /// </remarks>
    public static byte[] SerializeMatrix(Matrix<T> matrix)
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                WriteValue(writer, matrix[i, j]);
            }
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a matrix from a binary reader without specifying expected dimensions.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized data from.</param>
    /// <returns>The deserialized matrix with dimensions read from the binary data.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads a matrix (grid of numbers) from a file without needing to know
    /// its size in advance. The size information is stored in the file itself.
    /// </remarks>
    public static Matrix<T> DeserializeMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();
        Matrix<T> matrix = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = ReadValue(reader);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Deserializes a matrix from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized matrix.</param>
    /// <returns>The deserialized matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts a compact byte representation (which might have been
    /// stored in a database or received over a network) back into a usable matrix for AI calculations.
    /// </remarks>
    public static Matrix<T> DeserializeMatrix(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();
        Matrix<T> matrix = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = ReadValue(reader);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Serializes a tensor to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="tensor">The tensor to serialize.</param>
    /// <remarks>
    /// <para>
    /// This helper method writes a tensor's shape and values to a binary stream. It first writes the rank (number
    /// of dimensions), then each dimension size, and finally all the tensor values as doubles.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves a tensor's structure and values to a file.
    /// 
    /// When saving a tensor:
    /// - First, it saves how many dimensions the tensor has (its rank)
    /// - Then, it saves the size of each dimension
    /// - Finally, it saves all the actual values
    /// 
    /// This format ensures that when loading, the tensor can be reconstructed
    /// with exactly the same shape and values.
    /// </para>
    /// </remarks>
    public static void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }

        foreach (var value in tensor)
        {
            writer.Write(Convert.ToDouble(value));
        }
    }

    /// <summary>
    /// Serializes a vector to a binary format.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized data to.</param>
    /// <param name="vector">The vector to serialize.</param>
    /// <remarks>
    /// <b>For Beginners:</b> A vector is a one-dimensional array of numbers, like a single row or column
    /// of data. This method saves that list of numbers to a file so you can use it later.
    /// Vectors are commonly used in AI to represent features or weights.
    /// </remarks>
    public static void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            writer.Write(Convert.ToDouble(vector[i]));
        }
    }

    /// <summary>
    /// Deserializes a vector from a binary format with an expected length.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized data from.</param>
    /// <param name="length">The expected length of the vector.</param>
    /// <returns>The deserialized vector.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the stored vector length does not match the expected length.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads a previously saved vector (list of numbers) from a file,
    /// but checks that it has the length you expect. This is important because AI algorithms often
    /// need vectors of specific lengths to work correctly.
    /// </remarks>
    public static Vector<T> DeserializeVector(BinaryReader reader, int length)
    {
        int storedLength = reader.ReadInt32();

        if (storedLength != length)
        {
            throw new InvalidOperationException("Stored vector length does not match expected length.");
        }

        Vector<T> vector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        return vector;
    }

    /// <summary>
    /// Deserializes a tensor from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>The deserialized tensor.</returns>
    /// <remarks>
    /// <para>
    /// This helper method reads a tensor's shape and values from a binary stream. It first reads the rank (number
    /// of dimensions), then each dimension size, creates a tensor with that shape, and finally reads all the values.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a tensor's structure and values from a file.
    /// 
    /// When loading a tensor:
    /// - First, it reads how many dimensions the tensor has
    /// - Then, it reads the size of each dimension
    /// - It creates a new tensor with that shape
    /// - Finally, it reads all the values and fills the tensor
    /// 
    /// This process reverses the serialization process, reconstructing the tensor
    /// exactly as it was when saved.
    /// </para>
    /// </remarks>
    public static Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }

        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        return tensor;
    }

    /// <summary>
    /// Deserializes a vector from a binary reader without specifying expected length.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized data from.</param>
    /// <returns>The deserialized vector with length read from the binary data.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads a vector (list of numbers) from a file without needing to know
    /// its length in advance. The length information is stored in the file itself.
    /// </remarks>
    public static Vector<T> DeserializeVector(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        T[] array = new T[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = ReadValue(reader);
        }

        return new Vector<T>(array);
    }

    /// <summary>
    /// Serializes a vector to a byte array.
    /// </summary>
    /// <param name="vector">The vector to serialize.</param>
    /// <returns>A byte array containing the serialized vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts a vector (list of numbers) into a compact format
    /// that can be easily stored in memory or in a database. This is useful when you want to
    /// save trained model parameters for later use without retraining.
    /// </remarks>
    public static byte[] SerializeVector(Vector<T> vector)
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            WriteValue(writer, vector[i]);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a vector from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized vector.</param>
    /// <returns>The deserialized vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts a compact byte representation (which might have been
    /// stored in a database or received over a network) back into a usable vector for AI calculations.
    /// Vectors are essential in many AI algorithms for representing features, weights, or predictions.
    /// </remarks>
    public static Vector<T> DeserializeVector(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        int length = reader.ReadInt32();
        T[] array = new T[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = ReadValue(reader);
        }

        return new Vector<T>(array);
    }

    /// <summary>
    /// Serializes an interface instance by writing its type name to a BinaryWriter.
    /// </summary>
    /// <typeparam name="TInterface">The interface type to serialize.</typeparam>
    /// <param name="writer">The BinaryWriter to write the type name to.</param>
    /// <param name="instance">The interface instance to serialize.</param>
    /// <remarks>
    /// <para>
    /// This method writes the full name of the concrete type implementing the interface.
    /// If the instance is null, it writes an empty string.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves information about a specific part of your network.
    /// 
    /// It writes:
    /// - The name of the actual type used (if there is one)
    /// - An empty string if no specific type is used
    /// 
    /// This allows you to recreate the exact same setup when you load the network later.
    /// </para>
    /// </remarks>
    public static void SerializeInterface<TInterface>(BinaryWriter writer, TInterface? instance) where TInterface : class
    {
        writer.Write(instance?.GetType().FullName ?? string.Empty);
    }
}
