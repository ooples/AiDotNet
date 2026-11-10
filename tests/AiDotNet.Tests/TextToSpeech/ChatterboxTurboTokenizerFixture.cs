using System;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>The released Turbo tokenizer, pinned by repository revision and each file's SHA-256.</summary>
internal static class ChatterboxTurboTokenizerFixture
{
    private const string Revision = "749d1c1a46eb10492095d68fbcf55691ccf137cd";
    private static readonly Lazy<string> Resolved = new(Resolve);
    internal static string DirectoryPath => Resolved.Value;

    private static string Resolve()
    {
        var files = new[]
        {
            ("added_tokens.json", "72E4AB6ACB0D9309AC3DF4B526AE5FD80A2DA5BC5AB7BB02D85096A374F69193"),
            ("merges.txt", "1CE1664773C50F3E0CC8842619A93EDC4624525B728B188A9E0BE33B7726ADC5"),
            ("special_tokens_map.json", "92BA8063BF40AA163EADEBBFE0DE07C2AEBE44CF0D4A9E8726580B0781FD2640"),
            ("tokenizer_config.json", "BCA16A2AC1DDBD78B8D6228F0031884CC74B6EA54B967D6F6D2EBAE9CCDE23E6"),
            ("vocab.json", "F6BD25A65E4E63CA31360E9FB11C7E4F9A391A78385D640ACD814092DD6EEE4F"),
        };
        string? supplied = Environment.GetEnvironmentVariable("CHATTERBOX_TURBO_DIR");
        string directory = string.IsNullOrWhiteSpace(supplied)
            ? Path.Combine(Path.GetTempPath(), "aidotnet-published-chatterbox-turbo-tokenizer", Revision)
            : Path.GetFullPath(supplied!);
        Directory.CreateDirectory(directory);
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        foreach (var (name, hash) in files)
        {
            string path = Path.Combine(directory, name);
            if (File.Exists(path)) { Verify(path, hash); continue; }
            if (!string.IsNullOrWhiteSpace(supplied)) throw new FileNotFoundException("Released Turbo tokenizer asset is missing.", path);
            string temporary = path + "." + Guid.NewGuid().ToString("N");
            try
            {
                using (var stream = client.GetStreamAsync($"https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/{Revision}/{name}").GetAwaiter().GetResult())
                using (var output = File.Create(temporary)) stream.CopyTo(output);
                Verify(temporary, hash);
                try { File.Move(temporary, path); }
                catch (IOException) when (File.Exists(path)) { Verify(path, hash); }
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
        return directory;
    }

    private static void Verify(string path, string expected)
    {
        using var input = File.OpenRead(path);
        using var sha = SHA256.Create();
        string actual = BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "");
        if (!actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Released Turbo tokenizer asset hash differs for {path}: {actual}; expected {expected}.");
    }
}
