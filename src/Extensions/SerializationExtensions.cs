namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for serializing and deserializing data used in AI models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Serialization is the process of converting data structures or objects into a format 
/// that can be stored (in a file or database) or transmitted (across a network). Deserialization is the reverse process.
/// </para>
/// <para>
/// This class provides methods to save your AI model data to files and load them back later.
/// Think of it like saving your progress in a video game so you can continue later.
/// </para>
/// </remarks>
public static class SerializationExtensions
{
    /// <summary>
    /// Reads an array of type T from a binary stream.
    /// </summary>
    /// <typeparam name="T">The type of elements in the array.</typeparam>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>An array of type T containing the deserialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method reads an array of values from a file or stream. 
    /// It first reads how many items are in the array, then reads each item one by one.
    /// </para>
    /// <para>
    /// For example, if you saved an array of weights from your AI model, this method helps you load those weights back.
    /// </para>
    /// </remarks>
    public static T[] ReadArray<T>(this BinaryReader reader)
    {
        int length = reader.ReadInt32();
        T[] array = new T[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = (T)reader.ReadValue(typeof(T));
        }

        return array;
    }

    /// <summary>
    /// Writes an array of type T to a binary stream.
    /// </summary>
    /// <typeparam name="T">The type of elements in the array.</typeparam>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="array">The array to write.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method saves an array of values to a file or stream.
    /// It first writes how many items are in the array, then writes each item one by one.
    /// </para>
    /// <para>
    /// For example, if you want to save the weights of your AI model for later use, this method helps you do that.
    /// </para>
    /// </remarks>
    public static void WriteArray<T>(this BinaryWriter writer, T[] array)
    {
        writer.Write(array.Length);
        foreach (T item in array)
        {
            writer.WriteValue(Convert.ToDouble(item));
        }
    }

    /// <summary>
    /// Writes a value of a supported type to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="value">The value to write.</param>
    /// <exception cref="ArgumentException">Thrown when the value's type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method saves a single value to a file or stream.
    /// It handles different types of values (integers, decimals, true/false values) appropriately.
    /// </para>
    /// <para>
    /// Currently supported types are:
    /// <list type="bullet">
    ///   <item><description>int (whole numbers like 1, 2, 3)</description></item>
    ///   <item><description>double (decimal numbers with high precision like 3.14159265359)</description></item>
    ///   <item><description>float (decimal numbers with medium precision like 3.14)</description></item>
    ///   <item><description>bool (true/false values)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static void WriteValue(this BinaryWriter writer, object value)
    {
        switch (value)
        {
            case int i:
                writer.Write(i);
                break;
            case double d:
                writer.Write(d);
                break;
            case float f:
                writer.Write(f);
                break;
            case bool b:
                writer.Write(b);
                break;
            default:
                throw new ArgumentException($"Unsupported type: {value.GetType()}");
        }
    }

    /// <summary>
    /// Reads a value of the specified type from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="type">The type of value to read.</param>
    /// <returns>The deserialized value as an object.</returns>
    /// <exception cref="ArgumentException">Thrown when the specified type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method reads a single value from a file or stream.
    /// It handles different types of values based on what type you specify.
    /// </para>
    /// <para>
    /// Currently supported types are:
    /// <list type="bullet">
    ///   <item><description>int (whole numbers like 1, 2, 3)</description></item>
    ///   <item><description>double (decimal numbers with high precision like 3.14159265359)</description></item>
    ///   <item><description>float (decimal numbers with medium precision like 3.14)</description></item>
    ///   <item><description>bool (true/false values)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static object ReadValue(this BinaryReader reader, Type type)
    {
        if (type == typeof(int))
            return reader.ReadInt32();
        if (type == typeof(double))
            return reader.ReadDouble();
        if (type == typeof(float))
            return reader.ReadSingle();
        if (type == typeof(bool))
            return reader.ReadBoolean();
        throw new ArgumentException($"Unsupported type: {type}");
    }

    /// <summary>
    /// Reads an array of integers from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>An array of integers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is a specialized version that reads an array of whole numbers (integers)
    /// from a file or stream. It's optimized for integer arrays, which are commonly used in AI for things like
    /// neural network layer sizes, category indices, or feature counts.
    /// </para>
    /// <para>
    /// For example, if your neural network has layers with sizes [784, 128, 64, 10], this method could help you
    /// load that structure from a saved file.
    /// </para>
    /// </remarks>
    public static int[] ReadInt32Array(this BinaryReader reader)
    {
        int length = reader.ReadInt32();
        int[] array = new int[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = reader.ReadInt32();
        }

        return array;
    }

    /// <summary>
    /// Writes an array of integers to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="array">The array of integers to write.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is a specialized version that saves an array of whole numbers (integers)
    /// to a file or stream. It's optimized for integer arrays, which are commonly used in AI for things like
    /// neural network layer sizes, category indices, or feature counts.
    /// </para>
    /// <para>
    /// For example, if your neural network has layers with sizes [784, 128, 64, 10], this method could help you
    /// save that structure to a file.
    /// </para>
    /// </remarks>
    public static void WriteInt32Array(this BinaryWriter writer, int[] array)
    {
        writer.Write(array.Length);
        foreach (int value in array)
        {
            writer.Write(value);
        }
    }
}
