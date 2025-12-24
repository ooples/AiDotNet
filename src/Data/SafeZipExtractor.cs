using System.IO.Compression;

namespace AiDotNet.Data;

/// <summary>
/// Provides safe ZIP file extraction with protection against common security vulnerabilities.
/// </summary>
/// <remarks>
/// <para>This class provides protection against:</para>
/// <list type="bullet">
/// <item><description><b>Path Traversal Attacks</b>: Prevents extraction of entries with paths that escape the target directory (e.g., "../../../etc/passwd")</description></item>
/// <item><description><b>Zip Bombs</b>: Limits the total uncompressed size to prevent denial-of-service attacks</description></item>
/// </list>
/// </remarks>
internal static class SafeZipExtractor
{
    /// <summary>
    /// Default maximum uncompressed size (10 GB).
    /// </summary>
    private const long DefaultMaxUncompressedSize = 10L * 1024 * 1024 * 1024;

    /// <summary>
    /// Safely extracts a ZIP archive to the specified directory with security checks.
    /// </summary>
    /// <param name="zipFilePath">The path to the ZIP file to extract.</param>
    /// <param name="destinationDirectory">The directory to extract files to.</param>
    /// <param name="maxUncompressedSize">Maximum allowed total uncompressed size in bytes. Defaults to 10 GB.</param>
    /// <exception cref="ArgumentNullException">Thrown when zipFilePath or destinationDirectory is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the ZIP file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when a zip bomb is detected or path traversal is attempted.</exception>
    public static void ExtractToDirectory(string zipFilePath, string destinationDirectory, long maxUncompressedSize = DefaultMaxUncompressedSize)
    {
        if (string.IsNullOrEmpty(zipFilePath))
            throw new ArgumentNullException(nameof(zipFilePath));
        if (string.IsNullOrEmpty(destinationDirectory))
            throw new ArgumentNullException(nameof(destinationDirectory));
        if (!File.Exists(zipFilePath))
            throw new FileNotFoundException("ZIP file not found.", zipFilePath);

        // Ensure destination directory exists
        Directory.CreateDirectory(destinationDirectory);

        // Get the full path of the destination to use for path traversal checks
        string destinationFullPath = Path.GetFullPath(destinationDirectory);
        if (!destinationFullPath.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
        {
            destinationFullPath += Path.DirectorySeparatorChar;
        }

        using var archive = ZipFile.OpenRead(zipFilePath);

        // First pass: Check total uncompressed size and validate paths
        long totalUncompressedSize = 0;
        foreach (var entry in archive.Entries)
        {
            // Skip directory entries
            if (string.IsNullOrEmpty(entry.Name))
                continue;

            // Check for zip bomb
            totalUncompressedSize += entry.Length;
            if (totalUncompressedSize > maxUncompressedSize)
            {
                throw new InvalidDataException(
                    $"ZIP file exceeds maximum allowed uncompressed size of {maxUncompressedSize / (1024 * 1024)} MB. " +
                    "This may indicate a zip bomb attack.");
            }

            // Check for path traversal
            string entryFullPath = Path.GetFullPath(Path.Combine(destinationDirectory, entry.FullName));
            if (!entryFullPath.StartsWith(destinationFullPath, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"ZIP entry '{entry.FullName}' has a path that would extract outside the destination directory. " +
                    "This may indicate a path traversal attack.");
            }
        }

        // Second pass: Extract files
        foreach (var entry in archive.Entries)
        {
            // Skip directory entries
            if (string.IsNullOrEmpty(entry.Name))
                continue;

            string entryFullPath = Path.GetFullPath(Path.Combine(destinationDirectory, entry.FullName));

            // Create directory for the entry if it doesn't exist
            string? entryDirectory = Path.GetDirectoryName(entryFullPath);
            if (!string.IsNullOrEmpty(entryDirectory))
            {
                Directory.CreateDirectory(entryDirectory);
            }

            // Extract the entry
            entry.ExtractToFile(entryFullPath, overwrite: true);
        }
    }
}
