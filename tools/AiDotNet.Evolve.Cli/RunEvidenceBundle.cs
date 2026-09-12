using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Evolve.Cli;

/// <summary>Bounded, write-once evidence files; hashes prove integrity, not authenticity or scientific validity.</summary>
internal static class RunEvidenceBundle
{
    private const int MetadataLimit = 128 * 1024;
    private const int SourceLimit = 4 * 1024 * 1024;
    private static readonly string[] Required = { "configuration.json", "environment.json", "validation.json", "result.json" };
    private static readonly Encoding Utf8 = new UTF8Encoding(false, true);

    internal static void Create(string destination, IReadOnlyDictionary<string, byte[]> files, IEnumerable<string> prohibitedValues)
    {
        // Freeze before validation and hashing: caller mutation cannot change the bytes subsequently written.
        var copy = new SortedDictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var pair in files.Take(6))
        {
            ValidateNameAndLength(pair.Key, pair.Value.Length);
            copy.Add(pair.Key, (byte[])pair.Value.Clone());
        }
        RequireFiles(copy.Keys);
        string[] secrets = prohibitedValues.Where(value => !string.IsNullOrEmpty(value)).Distinct(StringComparer.Ordinal).ToArray();
        foreach (var pair in copy) ValidateContent(pair.Key, pair.Value, secrets);
        string target = Path.GetFullPath(destination);
        if (Directory.Exists(target) || File.Exists(target)) throw new IOException("Evidence destination must not already exist.");
        string parent = Path.GetDirectoryName(target) ?? throw new ArgumentException("An evidence destination needs a parent directory.");
        Directory.CreateDirectory(parent);
        string staging = Path.Combine(parent, ".aidotnet-export-" + Guid.NewGuid().ToString("N"));
        if (OperatingSystem.IsWindows()) Directory.CreateDirectory(staging);
        else Directory.CreateDirectory(staging, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        // Failed staging is deliberately retained for inspection. Never recursively delete an inferred user path.
        foreach (var pair in copy) WriteNew(Path.Combine(staging, pair.Key), pair.Value);
        byte[] manifest = JsonSerializer.SerializeToUtf8Bytes(new
        {
            SchemaVersion = 1,
            Files = copy.Select(pair => new { Name = pair.Key, Length = pair.Value.Length, Sha256 = Hash(pair.Value) }).ToArray()
        });
        WriteNew(Path.Combine(staging, "manifest.json"), manifest);
        Directory.Move(staging, target); // Same-parent publication is non-overwriting; an existing target is never replaced.
    }

    internal static IReadOnlyDictionary<string, byte[]> ReadAndVerify(string directory)
    {
        string root = Path.GetFullPath(directory);
        if ((File.GetAttributes(root) & FileAttributes.ReparsePoint) != 0) throw new IOException("Linked evidence roots are not supported.");
        byte[] manifest = ReadBounded(Path.Combine(root, "manifest.json"), MetadataLimit);
        using var document = ParseObject(manifest);
        JsonElement top = document.RootElement;
        if (top.EnumerateObject().Count() != 2 || top.GetProperty("SchemaVersion").GetInt32() != 1)
            throw new InvalidDataException("Unsupported evidence manifest.");
        JsonElement entries = top.GetProperty("Files");
        if (entries.ValueKind != JsonValueKind.Array || entries.GetArrayLength() is < 4 or > 5)
            throw new InvalidDataException("Invalid evidence file set.");
        var files = new SortedDictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (JsonElement item in entries.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object || item.EnumerateObject().Count() != 3)
                throw new InvalidDataException("Invalid evidence file entry.");
            string name = item.GetProperty("Name").GetString() ?? throw new InvalidDataException("Missing evidence name.");
            int length = item.GetProperty("Length").GetInt32();
            ValidateNameAndLength(name, length);
            byte[] data = ReadBounded(Path.Combine(root, name), name == "program.txt" ? SourceLimit : MetadataLimit);
            if (data.Length != length || !string.Equals(Hash(data), item.GetProperty("Sha256").GetString(), StringComparison.Ordinal))
                throw new InvalidDataException("Evidence integrity check failed.");
            ValidateContent(name, data, Array.Empty<string>());
            if (!files.TryAdd(name, data)) throw new InvalidDataException("Duplicate evidence file.");
        }
        RequireFiles(files.Keys);
        return files;
    }

    private static void RequireFiles(IEnumerable<string> names)
    {
        var set = new HashSet<string>(names, StringComparer.Ordinal);
        if (set.Count is < 4 or > 5 || Required.Any(name => !set.Contains(name)))
            throw new InvalidDataException("Evidence requires configuration, environment, validation and result documents.");
    }

    private static void ValidateNameAndLength(string name, int length)
    {
        if (name != "program.txt" && !Required.Contains(name, StringComparer.Ordinal))
            throw new InvalidDataException("Unknown evidence filename.");
        if (length < 1 || length > (name == "program.txt" ? SourceLimit : MetadataLimit))
            throw new InvalidDataException("Evidence file exceeds its size bound.");
    }

    private static void ValidateContent(string name, byte[] data, string[] secrets)
    {
        string text = Utf8.GetString(data);
        if (name == "program.txt") { CheckText(text, secrets); return; }
        using var document = ParseObject(data);
        Visit(document.RootElement, secrets);
    }

    private static JsonDocument ParseObject(byte[] data)
    {
        var document = JsonDocument.Parse(data, new JsonDocumentOptions { MaxDepth = 16 });
        try
        {
            if (document.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Evidence metadata must be a JSON object.");
            Visit(document.RootElement, Array.Empty<string>()); // Also reject duplicate manifest properties.
            return document;
        }
        catch { document.Dispose(); throw; }
    }

    private static void Visit(JsonElement item, string[] secrets)
    {
        if (item.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in item.EnumerateObject())
            {
                if (!names.Add(property.Name)) throw new InvalidDataException("Duplicate evidence metadata property.");
                CheckText(property.Name, secrets);
                Visit(property.Value, secrets);
            }
        }
        else if (item.ValueKind == JsonValueKind.Array)
            foreach (var child in item.EnumerateArray()) Visit(child, secrets);
        else CheckText(item.ValueKind == JsonValueKind.String ? item.GetString()! : item.GetRawText(), secrets);
    }

    private static void CheckText(string text, string[] secrets)
    {
        if (secrets.Any(secret => text.Contains(secret, StringComparison.Ordinal)))
            throw new InvalidDataException("Known configured secret detected; evidence was not published.");
    }

    private static byte[] ReadBounded(string path, int maximum)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Linked evidence files are not supported.");
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < 1 || stream.Length > maximum) throw new InvalidDataException("Evidence file exceeds its size bound.");
        var data = new byte[(int)stream.Length];
        stream.ReadExactly(data);
        if (stream.ReadByte() != -1) throw new InvalidDataException("Evidence changed while being read.");
        return data;
    }

    private static void WriteNew(string path, byte[] data)
    {
        var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write, Share = FileShare.None, Options = FileOptions.WriteThrough };
        if (!OperatingSystem.IsWindows()) options.UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;
        using var stream = new FileStream(path, options);
        stream.Write(data);
        stream.Flush(flushToDisk: true);
    }

    internal static string Hash(byte[] data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
}
