using System.Globalization;
using System.Security.Cryptography;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Downloads a front end's released asset once and checks it against the SHA-256 it was ported against.</summary>
internal static class VerifiedDownload
{
    /// <summary>The cache directory of front-end assets: <c>~/.aidotnet/models/<paramref name="name"/></c>.</summary>
    public static string CacheDirectory(string name) =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "models", name);

    /// <summary>Downloads <paramref name="url"/> to <paramref name="path"/> unless it is already there, failing (and
    /// leaving nothing behind) when the bytes are not the expected ones.</summary>
    public static void File(string url, string path, string sha256)
    {
        if (System.IO.File.Exists(path)) return;
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string part = path + ".part";
        try
        {
            if (System.IO.File.Exists(part)) System.IO.File.Delete(part);
            AiDotNet.Data.DatasetDownloader.DownloadFileAsync(url, part).GetAwaiter().GetResult();
            string hash = Sha256(part);
            if (!string.Equals(hash, sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"The download {url} has SHA-256 {hash}, not the released {sha256}.");
            System.IO.File.Move(part, path);
        }
        finally
        {
            if (System.IO.File.Exists(part)) System.IO.File.Delete(part);
        }
    }

    /// <summary>Downloads a zip archive and extracts it into <paramref name="directory"/> (at most
    /// <paramref name="maxUncompressedBytes"/>), unless every one of <paramref name="expected"/> files is already there.</summary>
    public static void Zip(string url, string directory, string sha256, long maxUncompressedBytes, params string[] expected)
    {
        if (expected.All(f => System.IO.File.Exists(Path.Combine(directory, f)))) return;
        string zip = Path.Combine(directory, Path.GetFileName(new Uri(url).AbsolutePath));
        File(url, zip, sha256);
        try
        {
            AiDotNet.Data.SafeZipExtractor.ExtractToDirectory(zip, directory, maxUncompressedBytes);
        }
        finally
        {
            System.IO.File.Delete(zip);
        }
        foreach (var f in expected)
            if (!System.IO.File.Exists(Path.Combine(directory, f))) throw new InvalidDataException($"The archive {url} has no {f}.");
    }

    /// <summary>A file's SHA-256 as lower-case hex.</summary>
    public static string Sha256(string path)
    {
        using var stream = System.IO.File.OpenRead(path);
        using var sha = SHA256.Create();
        return string.Concat(sha.ComputeHash(stream).Select(b => b.ToString("x2", CultureInfo.InvariantCulture)));
    }
}
