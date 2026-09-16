using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.TestImpact;

public enum AttributionRunMode { Collect, Discover, ExecutePlan }
public sealed record DiscoveryManifest(int Schema, string Workload, ExecutionContextIdentity Context, TestCaseIdentity[] Cases,
    string? ProfileJson = null);

public static class RunnerBinding
{
    // Hash every file, not just the test DLL: dependencies, native libraries,
    // settings and data files can change execution. Output must live elsewhere.
    // This binds a single immutable bundle; it is not GitHub provenance.
    public static string HashBundle(string directory)
    {
        string root = Path.GetFullPath(directory);
        if (!Directory.Exists(root)) throw new DirectoryNotFoundException(root);
        using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        int count = 0;
        foreach (string file in Files(root).OrderBy(path => Path.GetRelativePath(root, path).Replace('\\', '/'), StringComparer.Ordinal))
        {
            string relative = Path.GetRelativePath(root, file).Replace('\\', '/');
            byte[] name = Encoding.UTF8.GetBytes(relative);
            digest.AppendData(Encoding.ASCII.GetBytes(name.Length.ToString(System.Globalization.CultureInfo.InvariantCulture) + ":"));
            digest.AppendData(name);
            using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            digest.AppendData(SHA256.HashData(stream));
            count++;
        }
        if (count == 0) throw new InvalidDataException("An empty binary bundle cannot bind execution.");
        return Convert.ToHexStringLower(digest.GetHashAndReset());
    }

    private static IEnumerable<string> Files(string root)
    {
        if ((File.GetAttributes(root) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("Linked bundle paths are unsupported.");
        foreach (string child in Directory.EnumerateFileSystemEntries(root))
        {
            FileAttributes attributes = File.GetAttributes(child);
            if ((attributes & FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException("Linked bundle paths are unsupported.");
            if ((attributes & FileAttributes.Directory) != 0)
                foreach (string file in Files(child)) yield return file;
            else yield return child;
        }
    }

    public static void RequireOutsideBundle(string path, string bundle)
    {
        string relative = Path.GetRelativePath(Path.GetFullPath(bundle), Path.GetFullPath(path));
        if (!Path.IsPathRooted(relative) && relative != ".." &&
            !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new InvalidDataException("Plans and execution output must be outside the immutable binary bundle.");
    }

    public static string Serialize<T>(T value)
    {
        var options = new JsonSerializerOptions { WriteIndented = true };
        options.Converters.Add(new JsonStringEnumConverter(allowIntegerValues: false));
        return JsonSerializer.Serialize(value, options);
    }

    public static void WriteNew<T>(string path, T value, bool indented = true)
    {
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full) ?? throw new InvalidDataException("Missing output directory."));
        string pending = full + ".pending";
        var options = new JsonSerializerOptions { WriteIndented = indented };
        options.Converters.Add(new JsonStringEnumConverter(allowIntegerValues: false));
        using (var stream = new FileStream(pending, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            JsonSerializer.Serialize(stream, value, options);
        File.Move(pending, full, overwrite: false);
    }

    public static ExecutionPlan Prepare(DiscoveryManifest manifest, string[] selectedMethods, ValidationScope scope)
    {
        if (manifest.Schema != 1) throw new EvidenceException(EvidenceFailure.Format, "Unsupported discovery manifest.");
        return ExecutionEvidence.CreatePlan(manifest.Workload, manifest.Cases, selectedMethods, scope, manifest.Context);
    }
}
