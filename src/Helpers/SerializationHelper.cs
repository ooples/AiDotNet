using System.Runtime.Serialization.Formatters.Binary;

namespace AiDotNet.Helpers;

public static class SerializationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
                matrix[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        return matrix;
    }

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

    public static void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            writer.Write(Convert.ToDouble(vector[i]));
        }
    }

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
            vector[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return vector;
    }

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
}