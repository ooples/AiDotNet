namespace AiDotNet.Extensions;

public static class SerializationExtensions
{
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

    public static void WriteArray<T>(this BinaryWriter writer, T[] array)
    {
        writer.Write(array.Length);
        foreach (T item in array)
        {
            writer.WriteValue(Convert.ToDouble(item));
        }
    }

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
}
