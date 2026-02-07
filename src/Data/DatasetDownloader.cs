using System.Net.Http;
using System.Security.Cryptography;

namespace AiDotNet.Data;

/// <summary>
/// Shared utility for downloading, verifying, and extracting benchmark datasets.
/// </summary>
/// <remarks>
/// <para>
/// Provides common download-and-cache functionality used by all benchmark loaders.
/// Reuses <see cref="SafeZipExtractor"/> for secure archive extraction.
/// </para>
/// </remarks>
internal static class DatasetDownloader
{
    /// <summary>
    /// Gets the default cache directory for downloaded datasets.
    /// </summary>
    /// <param name="datasetName">Name of the dataset (used as subdirectory).</param>
    /// <returns>The full path to the dataset cache directory.</returns>
    public static string GetDefaultDataPath(string datasetName)
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".aidotnet",
            "datasets",
            datasetName);
    }

    /// <summary>
    /// Downloads a file from a URL if it doesn't already exist locally.
    /// </summary>
    /// <param name="url">The URL to download from.</param>
    /// <param name="destinationPath">The local file path to save to.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if the file was downloaded, false if it already existed.</returns>
    public static async Task<bool> DownloadFileAsync(
        string url,
        string destinationPath,
        CancellationToken cancellationToken = default)
    {
        if (File.Exists(destinationPath))
        {
            return false;
        }

        string? directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string tempPath = destinationPath + ".tmp";
        try
        {
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromMinutes(30);

            using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            using var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await response.Content.CopyToAsync(fileStream);

            // Move temp file to final location
            if (File.Exists(destinationPath))
            {
                File.Delete(destinationPath);
            }

            File.Move(tempPath, destinationPath);
            return true;
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); }
                catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Downloads a file and extracts it as a ZIP archive.
    /// </summary>
    /// <param name="url">The URL to download from.</param>
    /// <param name="extractDirectory">The directory to extract to.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task DownloadAndExtractZipAsync(
        string url,
        string extractDirectory,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(extractDirectory);

        string tempZip = Path.Combine(Path.GetTempPath(), $"aidotnet_{Guid.NewGuid()}.zip");
        try
        {
            await DownloadFileAsync(url, tempZip, cancellationToken);
            SafeZipExtractor.ExtractToDirectory(tempZip, extractDirectory);
        }
        finally
        {
            if (File.Exists(tempZip))
            {
                try { File.Delete(tempZip); }
                catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Downloads a tar.gz archive and extracts its contents to a directory.
    /// </summary>
    /// <param name="url">The URL to download from.</param>
    /// <param name="extractDirectory">The directory to extract to.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task DownloadAndExtractTarGzAsync(
        string url,
        string extractDirectory,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(extractDirectory);

        string tempFile = Path.Combine(Path.GetTempPath(), $"aidotnet_{Guid.NewGuid()}.tar.gz");
        try
        {
            await DownloadFileAsync(url, tempFile, cancellationToken);
            ExtractTarGz(tempFile, extractDirectory);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                try { File.Delete(tempFile); }
                catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Extracts a tar.gz file to the specified directory.
    /// </summary>
    /// <param name="tarGzPath">Path to the tar.gz file.</param>
    /// <param name="extractDirectory">Directory to extract to.</param>
    private static void ExtractTarGz(string tarGzPath, string extractDirectory)
    {
        using var fileStream = new FileStream(tarGzPath, FileMode.Open, FileAccess.Read);
        using var gzStream = new System.IO.Compression.GZipStream(
            fileStream, System.IO.Compression.CompressionMode.Decompress);

        // Read TAR entries manually (TAR format: 512-byte header blocks)
        byte[] header = new byte[512];
        while (true)
        {
            int bytesRead = ReadFull(gzStream, header, 0, 512);
            if (bytesRead < 512) break;

            // Check for zero block (end of archive)
            bool allZero = true;
            for (int i = 0; i < 512; i++)
            {
                if (header[i] != 0) { allZero = false; break; }
            }
            if (allZero) break;

            // Parse TAR header
            string name = ReadTarString(header, 0, 100);
            char typeFlag = (char)header[156];
            long size = ReadTarOctal(header, 124, 12);

            // Check for UStar prefix (extends name)
            string prefix = ReadTarString(header, 345, 155);
            if (prefix.Length > 0)
            {
                name = prefix + "/" + name;
            }

            // Validate path to prevent path traversal
            string fullPath = Path.GetFullPath(Path.Combine(extractDirectory, name));
            string normalizedExtractDir = Path.GetFullPath(extractDirectory);
            if (!fullPath.StartsWith(normalizedExtractDir, StringComparison.OrdinalIgnoreCase))
            {
                // Skip entries that would escape the extract directory
                SkipTarEntry(gzStream, size);
                continue;
            }

            if (typeFlag == '5' || name.EndsWith("/", StringComparison.Ordinal))
            {
                // Directory entry
                Directory.CreateDirectory(fullPath);
            }
            else if (typeFlag == '0' || typeFlag == '\0')
            {
                // Regular file
                string? dir = Path.GetDirectoryName(fullPath);
                if (!string.IsNullOrEmpty(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                using var output = new FileStream(fullPath, FileMode.Create, FileAccess.Write, FileShare.None);
                CopyBytes(gzStream, output, size);

                // TAR data is padded to 512-byte blocks
                long remainder = size % 512;
                if (remainder > 0)
                {
                    SkipBytes(gzStream, 512 - remainder);
                }
            }
            else
            {
                // Skip other entry types (symlinks, etc.)
                SkipTarEntry(gzStream, size);
            }
        }
    }

    private static int ReadFull(Stream stream, byte[] buffer, int offset, int count)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int read = stream.Read(buffer, offset + totalRead, count - totalRead);
            if (read == 0) break;
            totalRead += read;
        }
        return totalRead;
    }

    private static string ReadTarString(byte[] buffer, int offset, int length)
    {
        int end = offset;
        int limit = offset + length;
        while (end < limit && buffer[end] != 0) end++;
        return System.Text.Encoding.ASCII.GetString(buffer, offset, end - offset).Trim();
    }

    private static long ReadTarOctal(byte[] buffer, int offset, int length)
    {
        string octalStr = ReadTarString(buffer, offset, length);
        if (octalStr.Length == 0) return 0;
        return Convert.ToInt64(octalStr, 8);
    }

    private static void SkipTarEntry(Stream stream, long size)
    {
        if (size <= 0) return;
        // Round up to next 512-byte block boundary (TAR entries are block-aligned)
        long paddedSize = ((size + 511) / 512) * 512;
        SkipBytes(stream, paddedSize);
    }

    private static void SkipBytes(Stream stream, long count)
    {
        if (count <= 0) return;
        byte[] skipBuf = new byte[4096];
        long remaining = count;
        while (remaining > 0)
        {
            int toRead = (int)Math.Min(skipBuf.Length, remaining);
            int read = stream.Read(skipBuf, 0, toRead);
            if (read == 0) break;
            remaining -= read;
        }
    }

    private static void CopyBytes(Stream input, Stream output, long count)
    {
        byte[] buf = new byte[8192];
        long remaining = count;
        while (remaining > 0)
        {
            int toRead = (int)Math.Min(buf.Length, remaining);
            int read = input.Read(buf, 0, toRead);
            if (read == 0) break;
            output.Write(buf, 0, read);
            remaining -= read;
        }
    }

    /// <summary>
    /// Downloads a GZipped file and decompresses it.
    /// </summary>
    /// <param name="url">The URL to download from.</param>
    /// <param name="destinationPath">The local file path for the decompressed output.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task DownloadAndDecompressGzipAsync(
        string url,
        string destinationPath,
        CancellationToken cancellationToken = default)
    {
        if (File.Exists(destinationPath))
        {
            return;
        }

        string? directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string tempGz = destinationPath + ".gz.tmp";
        try
        {
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromMinutes(30);

            using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            using (var fileStream = new FileStream(tempGz, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await response.Content.CopyToAsync(fileStream);
            }

            // Decompress gzip
            using (var gzStream = new System.IO.Compression.GZipStream(
                new FileStream(tempGz, FileMode.Open, FileAccess.Read), System.IO.Compression.CompressionMode.Decompress))
            using (var outputStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await gzStream.CopyToAsync(outputStream);
            }
        }
        finally
        {
            if (File.Exists(tempGz))
            {
                try { File.Delete(tempGz); }
                catch { /* Ignore cleanup errors */ }
            }
        }
    }
}
