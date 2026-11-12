using System.IO.Compression;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released checkpoint's 512-rank GPT-2 byte BPE, offset and appended end token.</summary>
internal sealed class MetaVoiceTokenizer
{
    private const string End = "<|endoftext|>";
    private const string Pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    private readonly TiktokenEncoding _encoding;
    private readonly Dictionary<string, int> _byteRanks;
    internal int Offset { get; }
    internal int EndToken => Offset + 512;

    internal MetaVoiceTokenizer(IReadOnlyDictionary<byte[], int> ranks, int offset)
    {
        if (ranks.Count != 512 || (offset != 2049 && offset != 1025))
            throw new ArgumentException("Released MetaVoice requires 512 BPE ranks and the stage's actual token offset.");
        _encoding = new TiktokenEncoding(ranks, new[] { End }, TiktokenPattern.Gpt2); Offset = offset;
        _byteRanks = ranks.ToDictionary(pair => Convert.ToBase64String(pair.Key), pair => pair.Value, StringComparer.Ordinal);
    }

    internal Dictionary<string, int> ExportByteRanks() => new(_byteRanks, StringComparer.Ordinal);

    internal static MetaVoiceTokenizer FromByteRanks(IReadOnlyDictionary<string, int> ranks, int offset)
        => new(ranks.ToDictionary(pair => Convert.FromBase64String(pair.Key), pair => pair.Value), offset);

    internal int[] Encode(string text)
    {
        if (text.Contains(End)) throw new ArgumentException("The author tokenizer disallows explicit special-token text.", nameof(text));
        return _encoding.Encode(text, allowSpecial: false).Concat(new[] { 512 }).Select(id => id + Offset).ToArray();
    }

    internal string Decode(IEnumerable<int> ids) => _encoding.Decode(ids.Select(id => id - Offset));

    internal static Dictionary<string, object?> ReadMetadata(string checkpoint)
    {
        using var archive = ZipFile.OpenRead(checkpoint);
        var entries = archive.Entries.Where(entry => entry.FullName.EndsWith("/data.pkl", StringComparison.Ordinal) || entry.FullName == "data.pkl").ToArray();
        if (entries.Length != 1) throw new InvalidDataException("Expected one PyTorch checkpoint pickle entry.");
        using var entryStream = entries[0].Open();
        using var stream = new MemoryStream();
        entryStream.CopyTo(stream); stream.Position = 0;
        using var reader = new BinaryReader(stream);
        return new PickleParser(reader, preserveByteKeys: true).Parse() as Dictionary<string, object?>
            ?? throw new InvalidDataException("MetaVoice checkpoint metadata must be a dictionary.");
    }

    internal static MetaVoiceTokenizer FromCheckpoint(string checkpoint)
    {
        var root = ReadMetadata(checkpoint);
        var metadata = root["meta"] as Dictionary<string, object?> ?? throw new InvalidDataException("Missing MetaVoice meta dictionary.");
        var tokenizer = metadata["tokenizer"] as Dictionary<string, object?> ?? throw new InvalidDataException("Missing tokenizer metadata.");
        if ((string?)tokenizer["pat_str"] != Pattern) throw new InvalidDataException("MetaVoice tokenizer pre-tokenizer pattern differs.");
        var specials = tokenizer["special_tokens"] as Dictionary<string, object?> ?? throw new InvalidDataException("Missing tokenizer special tokens.");
        if (specials.Count != 1 || Convert.ToInt32(specials[End]) != 512) throw new InvalidDataException("MetaVoice end-token ID differs.");
        var ranks = tokenizer["mergeable_ranks"] as Dictionary<string, object?> ?? throw new InvalidDataException("Missing tokenizer byte ranks.");
        var result = new Dictionary<byte[], int>();
        foreach (var pair in ranks)
        {
            if (!pair.Key.StartsWith("bytes:", StringComparison.Ordinal)) throw new InvalidDataException("A tokenizer key lost its original bytes.");
            result.Add(Convert.FromBase64String(pair.Key.Substring(6)), Convert.ToInt32(pair.Value));
        }
        return new MetaVoiceTokenizer(result, Convert.ToInt32(tokenizer["offset"]));
    }
}
