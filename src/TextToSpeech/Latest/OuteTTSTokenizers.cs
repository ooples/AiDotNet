using System.IO.Compression;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// OuteTTS 1.0's tokenizers. OuteTTS-1.0-0.6B's (Apache-2.0) is built here: Qwen2's byte-level BPE (the vocabulary and
/// merges the library embeds), its NFC normalizer and split pattern, and the release's added tokens (exported by
/// <c>tools/reference-data/outetts_tokenizer_reference.py</c>). Llama-OuteTTS-1.0-1B's (CC-BY-NC-SA, as are its weights)
/// is read from the model's own <c>tokenizer.json</c>.
/// </summary>
internal static class OuteTTSTokenizers
{
    private const string Qwen2Pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private static readonly Lazy<ByteLevelBpeTokenizer> Qwen3Instance = new(BuildQwen3, isThreadSafe: true);

    /// <summary>OuteTTS-1.0-0.6B's tokenizer.</summary>
    public static ByteLevelBpeTokenizer Qwen3 => Qwen3Instance.Value;

    /// <summary>A tokenizer read from a model's <c>tokenizer.json</c> (Llama-OuteTTS-1.0-1B's, or a fine-tune's).</summary>
    public static ByteLevelBpeTokenizer FromFile(string path) => ByteLevelBpeTokenizer.FromJson(File.ReadAllText(path));

    private static string ReadResource(string name)
    {
        using var stream = typeof(OuteTTSTokenizers).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidDataException($"The embedded resource {name} is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, System.Text.Encoding.UTF8);
        return reader.ReadToEnd();
    }

    private static ByteLevelBpeTokenizer BuildQwen3()
    {
        var vocabulary = JsonConvert.DeserializeObject<Dictionary<string, int>>(ReadResource("AiDotNet.Tokenization.qwen2-vocab.json.gz"))
            ?? throw new InvalidDataException("The Qwen2 vocabulary is empty.");
        var merges = new Dictionary<string, int>(StringComparer.Ordinal);
        int rank = 0;
        foreach (var line in ReadResource("AiDotNet.Tokenization.qwen2-merges.txt.gz").Split('\n'))
        {
            var trimmed = line.TrimEnd('\r');
            if (trimmed.Length == 0 || trimmed.StartsWith("#version", StringComparison.Ordinal)) continue;
            int space = trimmed.IndexOf(' ');
            merges[trimmed.Substring(0, space) + "\u0001" + trimmed.Substring(space + 1)] = rank++;
        }
        var added = JArray.Parse(ReadResource("AiDotNet.TextToSpeech.Latest.outetts_qwen3_added_tokens.json.gz"))
            .Select(t => ((string)t[0]!, (int)t[1]!, (bool)t[2]!));
        return new ByteLevelBpeTokenizer(new Dictionary<string, int>(vocabulary, StringComparer.Ordinal), merges, added, Qwen2Pattern, nfc: true,
            ignoreMerges: false);
    }
}
