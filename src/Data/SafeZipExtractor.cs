using System.IO.Compression;

namespace AiDotNet.Data;

/// <summary>
/// Provides safe ZIP file extraction with protection against common security vulnerabilities.
/// </summary>
/// <remarks>
/// <para>This class provides protection against:</para>
/// <list type="bullet">
/// <item><description><b>Path Traversal Attacks</b>: Prevents extraction of entries with paths that escape the target directory (e.g., "../../../etc/passwd")</description></item>
/// <item><description><b>Zip Bombs</b>: Detects suspicious compression ratios and limits total uncompressed size during extraction</description></item>
/// <item><description><b>Entry Count Attacks</b>: Limits the number of entries to prevent inode exhaustion</description></item>
/// </list>
/// </remarks>
internal static class SafeZipExtractor
{
    /// <summary>
    /// Maximum number of entries allowed in an archive.
    /// </summary>
    private const int ThresholdEntries = 10000;

    /// <summary>
    /// Maximum total uncompressed size in bytes (1 GB).
    /// </summary>
    private const long ThresholdSize = 1L * 1024 * 1024 * 1024;

    /// <summary>
    /// Maximum allowed compression ratio before considering it a zip bomb.
    /// A ratio above 10 is highly suspicious.
    /// </summary>
    private const double ThresholdRatio = 10.0;

    /// <summary>
    /// Buffer size for reading compressed data.
    /// </summary>
    private const int BufferSize = 1024;

    /// <summary>
    /// Safely extracts a ZIP archive to the specified directory with security checks.
    /// </summary>
    /// <param name="zipFilePath">The path to the ZIP file to extract.</param>
    /// <param name="destinationDirectory">The directory to extract files to.</param>
    /// <param name="maxUncompressedSize">Maximum allowed total uncompressed size in bytes. Defaults to 1 GB.</param>
    /// <param name="maxEntries">Maximum number of entries allowed. Defaults to 10000.</param>
    /// <param name="maxCompressionRatio">Maximum compression ratio before considering it suspicious. Defaults to 10.</param>
    /// <exception cref="ArgumentNullException">Thrown when zipFilePath or destinationDirectory is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the ZIP file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when a zip bomb is detected or path traversal is attempted.</exception>
    public static void ExtractToDirectory(
        string zipFilePath,
        string destinationDirectory,
        long maxUncompressedSize = ThresholdSize,
        int maxEntries = ThresholdEntries,
        double maxCompressionRatio = ThresholdRatio)
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

        long totalSizeArchive = 0;
        int totalEntryArchive = 0;

        using var archive = ZipFile.OpenRead(zipFilePath);

        foreach (var entry in archive.Entries)
        {
            // Check entry count limit to prevent inode exhaustion
            totalEntryArchive++;
            if (totalEntryArchive > maxEntries)
            {
                throw new InvalidDataException(
                    $"ZIP file contains more than {maxEntries} entries. " +
                    "This may indicate an attack attempting to exhaust system inodes.");
            }

            // Skip directory entries
            if (string.IsNullOrEmpty(entry.Name))
                continue;

            // Check for path traversal before extraction
            string entryFullPath = Path.GetFullPath(Path.Combine(destinationDirectory, entry.FullName));
            if (!entryFullPath.StartsWith(destinationFullPath, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"ZIP entry '{entry.FullName}' has a path that would extract outside the destination directory. " +
                    "This may indicate a path traversal attack.");
            }

            // Create directory for the entry if it doesn't exist
            string? entryDirectory = Path.GetDirectoryName(entryFullPath);
            if (!string.IsNullOrEmpty(entryDirectory))
            {
                Directory.CreateDirectory(entryDirectory);
            }

            // Extract with compression ratio and size monitoring
            using (Stream entryStream = entry.Open())
            using (FileStream outputStream = new FileStream(entryFullPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                byte[] buffer = new byte[BufferSize];
                long totalSizeEntry = 0;
                int numBytesRead;

                while ((numBytesRead = entryStream.Read(buffer, 0, BufferSize)) > 0)
                {
                    totalSizeEntry += numBytesRead;
                    totalSizeArchive += numBytesRead;

                    // Check compression ratio - only if compressed length is known and > 0
                    if (entry.CompressedLength > 0)
                    {
                        double compressionRatio = (double)totalSizeEntry / entry.CompressedLength;
                        if (compressionRatio > maxCompressionRatio)
                        {
                            // Clean up the partial file
                            outputStream.Close();
                            try { File.Delete(entryFullPath); } catch { /* Ignore cleanup errors */ }

                            throw new InvalidDataException(
                                $"ZIP entry '{entry.FullName}' has a compression ratio of {compressionRatio:F1} " +
                                $"which exceeds the maximum allowed ratio of {maxCompressionRatio}. " +
                                "This may indicate a zip bomb attack.");
                        }
                    }

                    // Check total archive size
                    if (totalSizeArchive > maxUncompressedSize)
                    {
                        // Clean up the partial file
                        outputStream.Close();
                        try { File.Delete(entryFullPath); } catch { /* Ignore cleanup errors */ }

                        throw new InvalidDataException(
                            $"ZIP file exceeds maximum allowed uncompressed size of {maxUncompressedSize / (1024 * 1024)} MB. " +
                            "This may indicate a zip bomb attack.");
                    }

                    outputStream.Write(buffer, 0, numBytesRead);
                }
            }
        }
    }
}
