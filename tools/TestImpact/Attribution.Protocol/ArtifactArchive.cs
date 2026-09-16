using System.IO.Compression;

namespace AiDotNet.TestImpact;

public static class ArtifactArchive
{
    public static string ResolveContained(string root, string relative)
    {
        if (string.IsNullOrWhiteSpace(relative))
            throw new EvidenceException(EvidenceFailure.Format, "Artifact paths must be normalized relative paths.");
        string[] parts = relative.Split('/');
        if (Path.IsPathRooted(relative) || relative.IndexOfAny(['\\', ':', '<', '>', '"', '|', '?', '*']) >= 0 ||
            relative.Any(char.IsControl) || parts.Any(part => part is ".." or "." or "" || part.EndsWith(' ') || part.EndsWith('.') || Reserved(part)))
            throw new EvidenceException(EvidenceFailure.Format, "Artifact paths must be normalized relative paths.");
        string fullRoot = Path.GetFullPath(root);
        string path = Path.GetFullPath(Path.Combine(fullRoot, relative));
        if (!path.StartsWith(fullRoot + Path.DirectorySeparatorChar, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            throw new EvidenceException(EvidenceFailure.Format, "Artifact path escapes its directory.");
        return path;
    }

    private static bool Reserved(string part)
    {
        string stem = part.Split('.')[0].ToUpperInvariant();
        return stem is "CON" or "PRN" or "AUX" or "NUL" ||
            (stem.Length == 4 && (stem.StartsWith("COM", StringComparison.Ordinal) || stem.StartsWith("LPT", StringComparison.Ordinal)) && stem[3] is >= '1' and <= '9');
    }

    public static void Extract(string zip, string root)
    {
        if (Directory.Exists(root) || File.Exists(root)) throw new IOException("Artifact extraction destination must be new.");
        using ZipArchive archive = ZipFile.OpenRead(zip);
        long size = 0;
        if (archive.Entries.Count > 100_000) throw new InvalidDataException("Too many artifact entries.");
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var files = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var directories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        // Validate the complete layout before writing even the first file.
        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            size = checked(size + entry.Length);
            if (size > 2L * 1024 * 1024 * 1024 || (entry.ExternalAttributes >> 16 & 0xf000) == 0xa000)
                throw new InvalidDataException("Oversized or symbolic-link artifact entry.");
            string name = entry.FullName.TrimEnd('/');
            _ = ResolveContained(root, name);
            if (!names.Add(name)) throw new InvalidDataException("Duplicate artifact path.");
            bool directory = entry.FullName.EndsWith('/');
            if (files.Contains(name) || (!directory && directories.Contains(name)))
                throw new InvalidDataException("Artifact file/directory paths conflict.");
            if (directory) directories.Add(name);
            else files.Add(name);
            for (int separator = name.LastIndexOf('/'); separator >= 0; separator = name.LastIndexOf('/', separator - 1))
            {
                string parent = name[..separator];
                if (files.Contains(parent)) throw new InvalidDataException("Artifact file/directory paths conflict.");
                directories.Add(parent);
            }
        }
        Directory.CreateDirectory(root);
        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            string path = ResolveContained(root, entry.FullName.TrimEnd('/'));
            if (entry.FullName.EndsWith('/')) { Directory.CreateDirectory(path); continue; }
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? throw new IOException("Missing artifact parent."));
            using Stream input = entry.Open();
            using var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            byte[] buffer = new byte[81920];
            long written = 0;
            int read;
            while ((read = input.Read(buffer)) != 0)
            {
                written = checked(written + read);
                if (written > entry.Length) throw new InvalidDataException("Artifact entry exceeds its declared size.");
                output.Write(buffer, 0, read);
            }
            if (written != entry.Length) throw new InvalidDataException("Truncated artifact entry.");
        }
    }
}
