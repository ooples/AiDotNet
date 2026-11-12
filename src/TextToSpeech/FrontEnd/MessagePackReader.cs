using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The MessagePack that srsly writes (spaCy and Thinc model files, pkuseg's features): maps, arrays, strings, binaries,
/// integers, floats, booleans and nil. Values come back as <see cref="Dictionary{TKey,TValue}"/> (binary keys as their
/// ASCII text), <see cref="List{T}"/>, <see cref="string"/>, <c>byte[]</c>, <see cref="long"/> (unsigned 64-bit values above
/// <see cref="long.MaxValue"/> as <see cref="ulong"/>), <see cref="double"/>, <see cref="bool"/> or null.
/// </summary>
internal static class MessagePackReader
{
    /// <summary>Reads one value.</summary>
    public static object? Read(byte[] data)
    {
        int position = 0;
        return Value(data, ref position);
    }

    private static ulong BigEndian(byte[] d, ref int p, int bytes)
    {
        ulong v = 0;
        for (int i = 0; i < bytes; i++) v = (v << 8) | d[p++];
        return v;
    }

    private static string Text(byte[] d, ref int p, int length)
    {
        string s = Encoding.UTF8.GetString(d, p, length);
        p += length;
        return s;
    }

    private static byte[] Binary(byte[] d, ref int p, int length)
    {
        var bytes = new byte[length];
        System.Array.Copy(d, p, bytes, 0, length);
        p += length;
        return bytes;
    }

    private static object Unsigned(ulong v) => v <= long.MaxValue ? (long)v : v;

    private static object? Value(byte[] d, ref int p)
    {
        byte b = d[p++];
        if (b <= 0x7F) return (long)b;
        if (b >= 0xE0) return (long)(sbyte)b;
        if ((b & 0xF0) == 0x80) return Map(d, ref p, b & 0x0F);
        if ((b & 0xF0) == 0x90) return Array(d, ref p, b & 0x0F);
        if ((b & 0xE0) == 0xA0) return Text(d, ref p, b & 0x1F);
        switch (b)
        {
            case 0xC0: return null;
            case 0xC2: return false;
            case 0xC3: return true;
            case 0xC4: return Binary(d, ref p, (int)BigEndian(d, ref p, 1));
            case 0xC5: return Binary(d, ref p, (int)BigEndian(d, ref p, 2));
            case 0xC6: return Binary(d, ref p, (int)BigEndian(d, ref p, 4));
            case 0xCA:
            {
                var bytes = new byte[4];
                for (int i = 3; i >= 0; i--) bytes[i] = d[p++];
                return (double)BitConverter.ToSingle(bytes, 0);
            }
            case 0xCB: return BitConverter.Int64BitsToDouble((long)BigEndian(d, ref p, 8));
            case 0xCC: return (long)BigEndian(d, ref p, 1);
            case 0xCD: return (long)BigEndian(d, ref p, 2);
            case 0xCE: return (long)BigEndian(d, ref p, 4);
            case 0xCF: return Unsigned(BigEndian(d, ref p, 8));
            case 0xD0: return (long)(sbyte)BigEndian(d, ref p, 1);
            case 0xD1: return (long)(short)BigEndian(d, ref p, 2);
            case 0xD2: return (long)(int)BigEndian(d, ref p, 4);
            case 0xD3: return (long)BigEndian(d, ref p, 8);
            case 0xD9: return Text(d, ref p, (int)BigEndian(d, ref p, 1));
            case 0xDA: return Text(d, ref p, (int)BigEndian(d, ref p, 2));
            case 0xDB: return Text(d, ref p, (int)BigEndian(d, ref p, 4));
            case 0xDC: return Array(d, ref p, (int)BigEndian(d, ref p, 2));
            case 0xDD: return Array(d, ref p, (int)BigEndian(d, ref p, 4));
            case 0xDE: return Map(d, ref p, (int)BigEndian(d, ref p, 2));
            case 0xDF: return Map(d, ref p, (int)BigEndian(d, ref p, 4));
            default: throw new InvalidDataException($"Unsupported MessagePack type 0x{b:X2}.");
        }
    }

    private static List<object?> Array(byte[] d, ref int p, int count)
    {
        var list = new List<object?>(count);
        for (int i = 0; i < count; i++) list.Add(Value(d, ref p));
        return list;
    }

    private static Dictionary<object, object?> Map(byte[] d, ref int p, int count)
    {
        var map = new Dictionary<object, object?>(count);
        for (int i = 0; i < count; i++)
        {
            var key = Value(d, ref p) ?? throw new InvalidDataException("A MessagePack map key is nil.");
            if (key is byte[] bytes) key = Encoding.ASCII.GetString(bytes);
            map[key] = Value(d, ref p);
        }
        return map;
    }

    /// <summary>A NumPy array as srsly encodes it (a map of <c>nd</c>, <c>type</c>, <c>kind</c>, <c>shape</c>, <c>data</c>):
    /// its shape and little-endian float32 values.</summary>
    public static (int[] Shape, float[] Values) Float32Array(object? encoded)
    {
        var map = encoded as Dictionary<object, object?> ?? throw new InvalidDataException("Expected an encoded NumPy array.");
        var type = map["type"] is byte[] t ? Encoding.ASCII.GetString(t) : (string)map["type"]!;
        if (type != "<f4") throw new InvalidDataException($"Expected a float32 array, got {type}.");
        var shape = ((List<object?>)map["shape"]!).Select(v => Convert.ToInt32(v)).ToArray();
        var data = (byte[])map["data"]!;
        var values = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, values, 0, data.Length);
        return (shape, values);
    }
}
