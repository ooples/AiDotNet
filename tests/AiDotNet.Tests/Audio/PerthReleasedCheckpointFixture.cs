using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;

namespace AiDotNet.Tests.Audio;

/// <summary>The actual trained PerthNet distributed by Resemble in resemble-perth 1.0.1, with both archive and
/// extracted checkpoint pinned by SHA-256. The downloaded files stay in the machine's temporary test-asset cache.</summary>
internal static class PerthReleasedCheckpointFixture
{
    internal const string CheckpointSha256 = "a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670";
    private const string WheelSha256 = "65e9c37531b1a128a4a56226b75dece4521683cf3611b0b2a5ffe234f00c9342";
    private const string WheelUrl = "https://files.pythonhosted.org/packages/77/cc/73226dd776f8e9c2975f64f4efc22988fb37e5b185ba5cccd6f2e7196954/resemble_perth-1.0.1-py3-none-any.whl";
    private const string FileName = "perth_net_250000.pth.tar";
    private static readonly Lazy<string> Resolved = new(Resolve);

    internal static string CheckpointPath => Resolved.Value;

    private static string Resolve()
    {
        string? supplied = Environment.GetEnvironmentVariable("PERTH_CHECKPOINT");
        if (!string.IsNullOrWhiteSpace(supplied))
        {
            Verify(supplied, CheckpointSha256);
            return Path.GetFullPath(supplied);
        }
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-published-perth", CheckpointSha256);
        Directory.CreateDirectory(directory);
        string checkpoint = Path.Combine(directory, FileName);
        if (File.Exists(checkpoint)) { Verify(checkpoint, CheckpointSha256); return checkpoint; }
        string temporaryWheel = Path.Combine(directory, Guid.NewGuid().ToString("N") + ".whl");
        string temporaryCheckpoint = Path.Combine(directory, Guid.NewGuid().ToString("N") + ".pth");
        try
        {
            using (var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) })
            using (var response = client.GetAsync(WheelUrl, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
            {
                response.EnsureSuccessStatusCode();
                using var stream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
                using var output = File.Create(temporaryWheel);
                stream.CopyTo(output);
            }
            Verify(temporaryWheel, WheelSha256);
            using (var wheel = ZipFile.OpenRead(temporaryWheel))
            {
                var entry = wheel.Entries.Single(e => e.Name == FileName);
                using var input = entry.Open();
                using var output = File.Create(temporaryCheckpoint);
                input.CopyTo(output);
            }
            Verify(temporaryCheckpoint, CheckpointSha256);
            // Independent test hosts can fill the same immutable cache concurrently. Publish only a verified file.
            try { File.Move(temporaryCheckpoint, checkpoint); }
            catch (IOException) when (File.Exists(checkpoint)) { Verify(checkpoint, CheckpointSha256); }
            return checkpoint;
        }
        finally
        {
            if (File.Exists(temporaryWheel)) File.Delete(temporaryWheel);
            if (File.Exists(temporaryCheckpoint)) File.Delete(temporaryCheckpoint);
        }
    }

    private static void Verify(string path, string expected)
    {
        using var input = File.OpenRead(path);
        using var sha = SHA256.Create();
        string actual = BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "");
        if (!actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"The published Perth test checkpoint/archive SHA-256 differs: {actual}; expected {expected}.");
    }
}
