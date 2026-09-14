using System.Security.Cryptography;
using System.Text.Json;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace AiDotNet.Evolution.Deployment;

internal static class DeploymentEncoding
{
    internal static void RequireHash(string value)
    {
        if (value is null || value.Length != 64 || value.Any(c => !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f')))
            throw new ArgumentException("A lowercase SHA-256 digest is required.", nameof(value));
    }
    internal static void RequireLabel(string value, int maximum)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > maximum || value.Any(char.IsControl))
            throw new ArgumentException("A bounded non-control label is required.", nameof(value));
        _ = new System.Text.UTF8Encoding(false, true).GetByteCount(value);
    }
    internal static string Hash(byte[] bytes)
    {
        using var hash = SHA256.Create();
        return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", string.Empty).ToLowerInvariant();
    }
    internal static void RefuseLink(string path)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Linked deployment paths are not supported.");
    }
    internal static byte[] Read(string path, int maximum)
    {
        RefuseLink(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length <= 0 || stream.Length > maximum) throw new InvalidDataException("Invalid deployment file size.");
        var bytes = new byte[(int)stream.Length];
        int offset = 0;
        while (offset < bytes.Length)
        {
            int count = stream.Read(bytes, offset, bytes.Length - offset);
            if (count == 0) throw new EndOfStreamException();
            offset += count;
        }
        if (stream.ReadByte() != -1) throw new InvalidDataException("Deployment file changed while reading.");
        return bytes;
    }
    internal static void WriteNew(string path, byte[] bytes)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        stream.Write(bytes, 0, bytes.Length);
        stream.Flush(flushToDisk: true);
    }
    internal static T Parse<T>(byte[] bytes) where T : class
    {
        using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 16 });
        Unique(document.RootElement);
        return JsonSerializer.Deserialize<T>(bytes, new JsonSerializerOptions { MaxDepth = 16 })
            ?? throw new InvalidDataException("Missing deployment document.");
    }
    private static void Unique(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var item in value.EnumerateObject())
            {
                if (!names.Add(item.Name)) throw new InvalidDataException("Duplicate deployment property.");
                Unique(item.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (var item in value.EnumerateArray()) Unique(item);
    }
}
